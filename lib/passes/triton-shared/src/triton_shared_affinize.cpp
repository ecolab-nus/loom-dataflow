//===- triton_shared_affinize.cpp - Affinize Triton-shared indices -------===//
//
// This pass converts arithmetic index expressions in Triton-shared lowered
// kernels into affine.apply ops and replaces eligible memory ops with their
// affine counterparts. Affine map dims are restricted to the last three
// function arguments (grid/thread IDs) together with surrounding loop induction
// variables. All other function arguments are modeled as symbols. Unused dims
// and symbols are pruned from affine maps and their operand lists.
//
// Affinization is the entry point of the Triton → dataflow pipeline: it
// exposes the GPU-style indexing present in `tt.shared` kernels as canonical
// affine expressions so that later passes can wrap the program in
// `affine.parallel` loops, match hardware meshes described in the `df`
// dialect, and eventually enumerate legal mappings for spatial architectures.
//
// The transformation is conservative: only expressions that can be proven to be
// affine combinations of loop IVs, function arguments, and constants are
// converted.
//
//===----------------------------------------------------------------------===//

/**
 * @file triton_shared_affinize.cpp
 * @brief Affinize index computations and memory ops for Triton-shared kernels.
 * @details
 * Implementation outline
 * - Function argument normalization: promote 32-bit integer ABI args that
 *   participate in index arithmetic to `index` type, and rebuild nearby arith
 *   ops to operate on `index` to minimize casts.
 * - Dimension/symbol modeling: the last three function arguments and enclosing
 *   loop IVs are modeled as affine dims; other function arguments are modeled
 *   as affine symbols. Non-affine subgraphs are promoted to symbols.
 * - Affineization helpers:
 *   - LinearFormBuilder builds affine expressions with fixed dims while
 *     promoting unknowns to symbols.
 *   - Rewriters reconstruct `affine.apply`, `affine.min/max` around
 *     recognized arith patterns.
 * - Operation coverage:
 *   - `memref.load/store` → `affine.load/store` when all indices are affine.
 *   - `memref.reinterpret_cast`, `tensor.extract_slice`, `memref.subview`:
 *     rebuild selected operands (primarily offsets/sizes) with affine ops.
 *   - `scf.for`: rebuild with index-typed IV/bounds and affine-applied bounds.
 * - Cleanups: iterative DCE and redundant `arith.index_cast` removal.
 *
 * Constraints and limitations
 * - Only affine-preserving patterns are accepted: add/sub, mul by constant,
 *   min/max over affine expressions. Signed division is not converted to
 *   affine (to avoid trunc-vs-floor mismatch) and is promoted to a symbol.
 *   Anything else is treated as a symbol or left as-is.
 * - For `reinterpret_cast`, only offsets are aggressively affine-ized to avoid
 *   altering dominance-sensitive size/stride flows.
 * - Heuristics are conservative; failure to prove affinity leaves original IR
 *   unchanged (soundness over completeness).
 *
 * Usage
 * - Register the pass via `loom::passes::registerTritonSharedAffinizePass()`
 * - Invoke with `--loom-triton-shared-affinize` early in the pipeline before
 *   grid-to-parallel and spatial mapping passes.
 */

#include "triton_shared_affinize.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace loom {
namespace passes {

namespace {

/// Return true if the given affine expression is provably non-negative under
/// our conservative assumptions:
/// - Dim expressions are assumed non-negative (loop IVs, grid IDs).
/// - Constants are non-negative if their value >= 0.
/// - Symbols are assumed non-negative per pass policy.
/// - Add: non-negative if both sides are non-negative.
/// - Mul: only when multiplied by a non-negative constant and the other side is
///   non-negative. Multiplication by zero yields non-negative.
/// - Div/Mod/CeilDiv/FloorDiv inside the expression are treated as unknown.
///
/// \param expr Affine expression to analyze.
/// \return True iff the expression can be proven to evaluate to a value
///         greater than or equal to zero for all admissible dim/symbol values
///         under the assumptions above; false otherwise.
/// \note This predicate is intentionally conservative; when in doubt it
///       returns false to preserve correctness.
static bool isAffineExprProvenNonNegative(AffineExpr expr) {
  if (!expr)
    return false;
  switch (expr.getKind()) {
  case AffineExprKind::Constant: {
    auto c = llvm::cast<AffineConstantExpr>(expr);
    return c.getValue() >= 0;
  }
  case AffineExprKind::DimId:
    return true;
  case AffineExprKind::SymbolId:
    return true; // assume symbols are non-negative
  case AffineExprKind::Add: {
    auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
    return isAffineExprProvenNonNegative(bin.getLHS()) &&
           isAffineExprProvenNonNegative(bin.getRHS());
  }
  case AffineExprKind::Mul: {
    auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
    // One side must be a constant.
    if (auto lc = llvm::dyn_cast<AffineConstantExpr>(bin.getLHS()))
      return (lc.getValue() >= 0) &&
             (lc.getValue() == 0 ||
              isAffineExprProvenNonNegative(bin.getRHS()));
    if (auto rc = llvm::dyn_cast<AffineConstantExpr>(bin.getRHS()))
      return (rc.getValue() >= 0) &&
             (rc.getValue() == 0 ||
              isAffineExprProvenNonNegative(bin.getLHS()));
    return false;
  }
  case AffineExprKind::Mod:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::FloorDiv:
    return false;
  }
  return false;
}

/// Builder that constructs an affine expression using fixed dims (the three
/// spatial arguments) and promotes any non-affine sub-expressions to symbols.
/**
 * @brief Builds affine expressions from SSA values with fixed dim operands.
 *
 * @details Converts arithmetic index computations rooted at an SSA `Value`
 * into an `AffineExpr` using a fixed set of dim operands (grid IDs and
 * surrounding loop IVs). Sub-expressions that cannot be proven affine under
 * the pass' restrictions are promoted to symbols. The builder memoizes visited
 * nodes to form a DAG and optionally records error conditions for unsupported
 * constructs (e.g., signed division by non-constant).
 */
class LinearFormBuilder {
public:
  /// Construct a builder that treats `fixedDims` as affine dims and promotes
  /// other leaf values to symbols. If `errorFlag` is provided, it will be set
  /// to true when encountering expressions that violate the builder's affine
  /// restrictions (e.g., div by non-positive constant).
  LinearFormBuilder(MLIRContext *ctx, ArrayRef<Value> fixedDims,
                    bool *errorFlag = nullptr)
      : ctx(ctx), errorFlag(errorFlag) {
    dims.assign(fixedDims.begin(), fixedDims.end());
    for (unsigned i = 0; i < dims.size(); ++i)
      dimIndex.try_emplace(dims[i], i);
  }

  /// Build an affine expression from `v`.
  ///
  /// - Direct dim values map to `dim#i`.
  /// - Constants become `AffineConstantExpr`.
  /// - add/sub compose recursively; unknown sides become symbols.
  /// - mul is allowed only when one side is a constant; otherwise a symbol.
  /// - signed div lowers to floorDiv by positive constant if the numerator is
  ///   provably non-negative; otherwise a symbol and error is reported.
  /// - Any other node turns into a fresh symbol (with an inserted index cast
  ///   when needed).
  ///
  /// @param v Root SSA value of the arithmetic expression.
  /// @return Affine expression modeling `v` using current dims/symbols, or an
  ///         empty expression when the builder chooses to defer to callers for
  ///         partial handling.
  AffineExpr build(Value v) {
    if (auto it = cache.find(v); it != cache.end())
      return it->second;

    // Direct dims.
    if (auto it2 = dimIndex.find(v); it2 != dimIndex.end())
      return cache[v] = getAffineDimExpr(it2->second, ctx);

    // Allow index_cast passthrough.
    if (auto castOp = dyn_cast_or_null<arith::IndexCastOp>(v.getDefiningOp()))
      return cache[v] = build(castOp.getIn());

    // Constants.
    if (auto cidx = dyn_cast_or_null<arith::ConstantIndexOp>(v.getDefiningOp()))
      return cache[v] = getAffineConstantExpr(cidx.value(), ctx);
    if (auto cint = dyn_cast_or_null<arith::ConstantIntOp>(v.getDefiningOp()))
      return cache[v] = getAffineConstantExpr(cint.value(), ctx);

    // Addition/subtraction: linearize each side; if a side fails, treat it as a
    // symbol operand.
    if (auto addi = dyn_cast_or_null<arith::AddIOp>(v.getDefiningOp())) {
      AffineExpr a = build(addi.getLhs());
      AffineExpr b = build(addi.getRhs());
      if (!a)
        a = getOrAddSymbol(addi.getLhs());
      if (!b)
        b = getOrAddSymbol(addi.getRhs());
      return cache[v] = a + b;
    }
    if (auto subi = dyn_cast_or_null<arith::SubIOp>(v.getDefiningOp())) {
      AffineExpr a = build(subi.getLhs());
      AffineExpr b = build(subi.getRhs());
      if (!a)
        a = getOrAddSymbol(subi.getLhs());
      if (!b)
        b = getOrAddSymbol(subi.getRhs());
      return cache[v] = a - b;
    }

    // Multiplication: allow multiply by constant; otherwise promote whole node
    // to a symbol.
    if (auto muli = dyn_cast_or_null<arith::MulIOp>(v.getDefiningOp())) {
      AffineExpr a = build(muli.getLhs());
      AffineExpr b = build(muli.getRhs());
      if (auto ca = dyn_cast<AffineConstantExpr>(a))
        return cache[v] = b ? b * ca.getValue() : getOrAddSymbol(v);
      if (auto cb = dyn_cast<AffineConstantExpr>(b))
        return cache[v] = a ? a * cb.getValue() : getOrAddSymbol(v);
      return cache[v] = getOrAddSymbol(v);
    }

    // Signed division: if denominator is a positive constant and the numerator
    // is provably non-negative, lower to affine floorDiv. Otherwise, emit an
    // error and treat as a symbol.
    if (auto divi = dyn_cast_or_null<arith::DivSIOp>(v.getDefiningOp())) {
      AffineExpr a = build(divi.getLhs());
      AffineExpr b = build(divi.getRhs());
      if (auto cb = dyn_cast<AffineConstantExpr>(b)) {
        int64_t denom = cb.getValue();
        if (denom > 0 && a && isAffineExprProvenNonNegative(a))
          return cache[v] = a.floorDiv(denom);
        // Error cases: zero/non-positive denom or numerator not proven >= 0.
        if (Operation *op = divi.getOperation())
          op->emitError("signed division not supported: denominator must be a "
                        "positive constant and numerator must be non-negative");
        if (errorFlag)
          *errorFlag = true;
      } else {
        if (Operation *op = divi.getOperation())
          op->emitError("signed division by non-constant is not supported in "
                        "affine lowering");
        if (errorFlag)
          *errorFlag = true;
      }
      return cache[v] = getOrAddSymbol(v);
    }

    // Block arguments not among fixed dims are symbols (including loop IVs).
    if (isa<BlockArgument>(v))
      return cache[v] = getOrAddSymbol(v);

    // Default: promote to symbol.
    return cache[v] = getOrAddSymbol(v);
  }

  /// Return the fixed dim operands used by this builder.
  ArrayRef<Value> getDims() const { return dims; }
  /// Return the symbol operands discovered while building expressions.
  ArrayRef<Value> getSymbols() const { return symbols; }

private:
  /// Obtain a symbol expression for `v`, inserting an `arith.index_cast` to
  /// index type when necessary and recording `v` in the symbol list if it is
  /// new.
  AffineExpr getOrAddSymbol(Value v) {
    auto it = symIndex.find(v);
    if (it != symIndex.end())
      return getAffineSymbolExpr(it->second, ctx);
    // Ensure index type for symbol operands.
    if (!v.getType().isIndex()) {
      if (auto ic = dyn_cast_or_null<arith::IndexCastOp>(v.getDefiningOp())) {
        v = ic.getResult();
      } else if (Operation *def = v.getDefiningOp()) {
        // Cast after defining op.
        OpBuilder tb(def);
        tb.setInsertionPointAfter(def);
        v = tb.create<arith::IndexCastOp>(def->getLoc(), IndexType::get(ctx), v)
                .getResult();
      } else if (auto barg = dyn_cast<BlockArgument>(v)) {
        // Insert at function entry if this is a function argument.
        if (auto func =
                dyn_cast<func::FuncOp>(barg.getOwner()->getParentOp())) {
          OpBuilder tb(func.getBody());
          tb.setInsertionPointToStart(&func.front());
          v = tb.create<arith::IndexCastOp>(func.getLoc(), IndexType::get(ctx),
                                            v)
                  .getResult();
        }
      }
    }
    unsigned idx = symbols.size();
    symbols.push_back(v);
    symIndex.try_emplace(v, idx);
    return getAffineSymbolExpr(idx, ctx);
  }

  MLIRContext *ctx;
  SmallVector<Value, 6> dims;
  SmallVector<Value, 8> symbols;
  DenseMap<Value, unsigned> dimIndex;
  DenseMap<Value, unsigned> symIndex;
  DenseMap<Value, AffineExpr> cache;
  bool *errorFlag = nullptr;
};

/// Collect symbol indices used by an affine expression.
/// Collect the set of symbol positions referenced by an affine expression.
///
/// @param expr Affine expression to inspect.
/// @param positions Output container appended with all `symbol#i` positions in
///        first-seen order.
static void collectUsedSymbolPositions(AffineExpr expr,
                                       SmallVectorImpl<unsigned> &positions) {
  if (!expr)
    return;
  AffineExprKind k = expr.getKind();
  if (k == AffineExprKind::SymbolId) {
    auto sym = llvm::cast<AffineSymbolExpr>(expr);
    positions.push_back(sym.getPosition());
    return;
  }
  if (k == AffineExprKind::DimId || k == AffineExprKind::Constant)
    return;
  if (k == AffineExprKind::Add || k == AffineExprKind::Mul ||
      k == AffineExprKind::Mod || k == AffineExprKind::FloorDiv ||
      k == AffineExprKind::CeilDiv) {
    auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
    collectUsedSymbolPositions(bin.getLHS(), positions);
    collectUsedSymbolPositions(bin.getRHS(), positions);
    return;
  }
}

/// Remap symbol ids in `expr` to a compact 0..k-1 set in first-use order,
/// returning the remapped expression and filling `orderedSymbolValues` with the
/// corresponding Values from `allSymbolValues`.
/// Remap symbol ids in `expr` to a compact 0..k-1 domain in first-use order.
///
/// @param expr Expression whose symbols will be remapped.
/// @param orderedSymbolValues Output list of `Value`s corresponding to the new
///        symbol ordering.
/// @param allSymbolValues Original symbol `Value`s indexed by old positions.
/// @return Expression with symbols rewritten to the new compact ordering.
static AffineExpr
remapSymbolsSequential(AffineExpr expr,
                       SmallVectorImpl<Value> &orderedSymbolValues,
                       ArrayRef<Value> allSymbolValues) {
  if (!expr)
    return expr;
  DenseMap<unsigned, unsigned> oldToNew;
  unsigned next = 0;
  std::function<AffineExpr(AffineExpr)> remap =
      [&](AffineExpr e) -> AffineExpr {
    switch (e.getKind()) {
    case AffineExprKind::SymbolId: {
      auto sym = llvm::cast<AffineSymbolExpr>(e);
      unsigned oldPos = sym.getPosition();
      auto it = oldToNew.find(oldPos);
      unsigned newPos;
      if (it == oldToNew.end()) {
        newPos = next++;
        oldToNew.try_emplace(oldPos, newPos);
        orderedSymbolValues.push_back(allSymbolValues[oldPos]);
      } else {
        newPos = it->second;
      }
      return getAffineSymbolExpr(newPos, e.getContext());
    }
    case AffineExprKind::DimId:
    case AffineExprKind::Constant:
      return e;
    case AffineExprKind::Add: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()) + remap(bin.getRHS());
    }
    case AffineExprKind::Mul: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()) * remap(bin.getRHS());
    }
    case AffineExprKind::Mod: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return getAffineBinaryOpExpr(AffineExprKind::Mod, remap(bin.getLHS()),
                                   remap(bin.getRHS()));
    }
    case AffineExprKind::FloorDiv: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return getAffineBinaryOpExpr(AffineExprKind::FloorDiv,
                                   remap(bin.getLHS()), remap(bin.getRHS()));
    }
    case AffineExprKind::CeilDiv: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return getAffineBinaryOpExpr(AffineExprKind::CeilDiv, remap(bin.getLHS()),
                                   remap(bin.getRHS()));
    }
    }
    return e;
  };
  return remap(expr);
}

/// Remap symbol ids in `expr` to a common ordering provided by `oldToNew`.
/// The keys of `oldToNew` are positions in `allSymbolValues`. The values are
/// the new compact positions [0..k-1].
/// Remap symbol ids in `expr` using an explicit old->new mapping.
///
/// @param expr Expression whose symbols will be remapped.
/// @param oldToNew Mapping from old symbol positions to new positions.
/// @return Expression with symbol positions rewritten.
static AffineExpr
remapSymbolsWithMapping(AffineExpr expr,
                        const DenseMap<unsigned, unsigned> &oldToNew) {
  if (!expr)
    return expr;
  MLIRContext *ctx = expr.getContext();
  std::function<AffineExpr(AffineExpr)> remap =
      [&](AffineExpr e) -> AffineExpr {
    switch (e.getKind()) {
    case AffineExprKind::SymbolId: {
      auto sym = llvm::cast<AffineSymbolExpr>(e);
      unsigned oldPos = sym.getPosition();
      auto it = oldToNew.find(oldPos);
      assert(it != oldToNew.end() && "symbol position not in mapping");
      return getAffineSymbolExpr(it->second, ctx);
    }
    case AffineExprKind::DimId:
    case AffineExprKind::Constant:
      return e;
    case AffineExprKind::Add: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()) + remap(bin.getRHS());
    }
    case AffineExprKind::Mul: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()) * remap(bin.getRHS());
    }
    case AffineExprKind::Mod: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return getAffineBinaryOpExpr(AffineExprKind::Mod, remap(bin.getLHS()),
                                   remap(bin.getRHS()));
    }
    case AffineExprKind::FloorDiv: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return getAffineBinaryOpExpr(AffineExprKind::FloorDiv,
                                   remap(bin.getLHS()), remap(bin.getRHS()));
    }
    case AffineExprKind::CeilDiv: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return getAffineBinaryOpExpr(AffineExprKind::CeilDiv, remap(bin.getLHS()),
                                   remap(bin.getRHS()));
    }
    }
    return e;
  };
  return remap(expr);
}

/// Collect dim indices used by an affine expression.
/// Collect the set of dim positions referenced by an affine expression.
///
/// @param expr Affine expression to inspect.
/// @param positions Output container appended with all `dim#i` positions in
///        first-seen order.
static void collectUsedDimPositions(AffineExpr expr,
                                    SmallVectorImpl<unsigned> &positions) {
  if (!expr)
    return;
  AffineExprKind k = expr.getKind();
  if (k == AffineExprKind::DimId) {
    auto dim = llvm::cast<AffineDimExpr>(expr);
    positions.push_back(dim.getPosition());
    return;
  }
  if (k == AffineExprKind::SymbolId || k == AffineExprKind::Constant)
    return;
  if (k == AffineExprKind::Add || k == AffineExprKind::Mul ||
      k == AffineExprKind::Mod || k == AffineExprKind::FloorDiv ||
      k == AffineExprKind::CeilDiv) {
    auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
    collectUsedDimPositions(bin.getLHS(), positions);
    collectUsedDimPositions(bin.getRHS(), positions);
    return;
  }
}

/// Remap dim ids in `expr` to a common ordering provided by `oldToNew`.
/// The keys of `oldToNew` are positions in the original dims. The values are
/// the new compact positions [0..k-1].
/// Remap dim ids in `expr` using an explicit old->new mapping.
///
/// @param expr Expression whose dims will be remapped.
/// @param oldToNew Mapping from old dim positions to new positions.
/// @return Expression with dim positions rewritten.
static AffineExpr
remapDimsWithMapping(AffineExpr expr,
                     const DenseMap<unsigned, unsigned> &oldToNew) {
  if (!expr)
    return expr;
  MLIRContext *ctx = expr.getContext();
  std::function<AffineExpr(AffineExpr)> remap =
      [&](AffineExpr e) -> AffineExpr {
    switch (e.getKind()) {
    case AffineExprKind::DimId: {
      auto dim = llvm::cast<AffineDimExpr>(e);
      unsigned oldPos = dim.getPosition();
      auto it = oldToNew.find(oldPos);
      assert(it != oldToNew.end() && "dim position not in mapping");
      return getAffineDimExpr(it->second, ctx);
    }
    case AffineExprKind::SymbolId:
    case AffineExprKind::Constant:
      return e;
    case AffineExprKind::Add: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()) + remap(bin.getRHS());
    }
    case AffineExprKind::Mul: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()) * remap(bin.getRHS());
    }
    case AffineExprKind::Mod: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return getAffineBinaryOpExpr(AffineExprKind::Mod, remap(bin.getLHS()),
                                   remap(bin.getRHS()));
    }
    case AffineExprKind::FloorDiv: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return getAffineBinaryOpExpr(AffineExprKind::FloorDiv,
                                   remap(bin.getLHS()), remap(bin.getRHS()));
    }
    case AffineExprKind::CeilDiv: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return getAffineBinaryOpExpr(AffineExprKind::CeilDiv, remap(bin.getLHS()),
                                   remap(bin.getRHS()));
    }
    }
    return e;
  };
  return remap(expr);
}

/// Remap both dims and symbols to 0..k-1 sequential, returning the remapped
/// expression and filling ordered dims/symbols in first-use order.
/// Remap both dims and symbols to compact 0..k-1 domains in first-use order.
///
/// @param expr Expression whose dims and symbols will be remapped.
/// @param orderedDims Output list of dim `Value`s in remapped order.
/// @param allDims Original dim `Value`s indexed by old positions.
/// @param orderedSymbols Output list of symbol `Value`s in remapped order.
/// @param allSymbols Original symbol `Value`s indexed by old positions.
/// @return Expression with dim and symbol positions rewritten.
static AffineExpr remapDimsAndSymbolsSequential(
    AffineExpr expr, SmallVectorImpl<Value> &orderedDims,
    ArrayRef<Value> allDims, SmallVectorImpl<Value> &orderedSymbols,
    ArrayRef<Value> allSymbols) {
  if (!expr)
    return expr;
  DenseMap<unsigned, unsigned> oldDimToNew;
  DenseMap<unsigned, unsigned> oldSymToNew;
  unsigned nextDim = 0, nextSym = 0;
  MLIRContext *ctx = expr.getContext();
  std::function<AffineExpr(AffineExpr)> remap =
      [&](AffineExpr e) -> AffineExpr {
    switch (e.getKind()) {
    case AffineExprKind::DimId: {
      auto d = llvm::cast<AffineDimExpr>(e);
      unsigned oldPos = d.getPosition();
      unsigned newPos;
      if (auto it = oldDimToNew.find(oldPos); it != oldDimToNew.end())
        newPos = it->second;
      else {
        newPos = nextDim++;
        oldDimToNew.try_emplace(oldPos, newPos);
        orderedDims.push_back(allDims[oldPos]);
      }
      return getAffineDimExpr(newPos, ctx);
    }
    case AffineExprKind::SymbolId: {
      auto s = llvm::cast<AffineSymbolExpr>(e);
      unsigned oldPos = s.getPosition();
      unsigned newPos;
      if (auto it = oldSymToNew.find(oldPos); it != oldSymToNew.end())
        newPos = it->second;
      else {
        newPos = nextSym++;
        oldSymToNew.try_emplace(oldPos, newPos);
        orderedSymbols.push_back(allSymbols[oldPos]);
      }
      return getAffineSymbolExpr(newPos, ctx);
    }
    case AffineExprKind::Constant:
      return e;
    case AffineExprKind::Add: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()) + remap(bin.getRHS());
    }
    case AffineExprKind::Mul: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return remap(bin.getLHS()) * remap(bin.getRHS());
    }
    case AffineExprKind::Mod: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return getAffineBinaryOpExpr(AffineExprKind::Mod, remap(bin.getLHS()),
                                   remap(bin.getRHS()));
    }
    case AffineExprKind::FloorDiv: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return getAffineBinaryOpExpr(AffineExprKind::FloorDiv,
                                   remap(bin.getLHS()), remap(bin.getRHS()));
    }
    case AffineExprKind::CeilDiv: {
      auto bin = llvm::cast<AffineBinaryOpExpr>(e);
      return getAffineBinaryOpExpr(AffineExprKind::CeilDiv, remap(bin.getLHS()),
                                   remap(bin.getRHS()));
    }
    }
    return e;
  };
  return remap(expr);
}

/**
 * @brief Pass that converts Triton-shared index arithmetic to affine IR.
 *
 * @details Operates on each `func::FuncOp` within a `ModuleOp` and:
 *   - Promotes 32-bit integer ABI arguments to `index` type.
 *   - Canonicalizes arithmetic to operate on `index`.
 *   - Treats the last three function arguments and surrounding loop IVs as
 *     affine dims; other function arguments and non-affine values become
 *     symbols.
 *   - Attempts to rebuild index expressions as `affine.apply`, and converts
 *     `memref.load/store` to their affine counterparts when all indices are
 *     affine.
 *   - Rewrites selected operands of `memref.reinterpret_cast`,
 *     `tensor.extract_slice`, and `memref.subview` using affine ops.
 *   - Rebuilds `scf.for` with index-typed IV/bounds and affine-applied bounds.
 *   - Performs iterative DCE and redundant cast removal.
 *
 * Errors in affineization (e.g., unsupported signed division) are recorded and
 * reported; the pass signals failure if any were encountered.
 */
class TritonSharedAffinizePass
    : public PassWrapper<TritonSharedAffinizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonSharedAffinizePass)

  /// Return the command-line argument to invoke this pass.
  StringRef getArgument() const override {
    return "loom-triton-shared-affinize";
  }
  /// Return a human-readable description of the pass.
  StringRef getDescription() const override {
    return "Convert Triton-shared index arithmetic to affine.apply and affine "
           "memops."
           " Dims are the last three function args and surrounding loop IVs;"
           " other function args are symbols. Unused dims/symbols are pruned.";
  }

  /// Declare dependent dialects required by this pass.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect, func::FuncDialect, scf::SCFDialect,
                    linalg::LinalgDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect>();
  }

  /**
   * @brief Execute the pass on the target `ModuleOp`.
   *
   * @details For each `func::FuncOp`:
   *   1) Promote i32 arguments to `index` and canonicalize arithmetic to index.
   *   2) Identify dims (last three function args + surrounding loop IVs).
   *   3) Attempt to rebuild index expressions with `affine.apply` using a
   *      `LinearFormBuilder`, promoting unknowns to symbols.
   *   4) Convert eligible loads/stores to `affine::AffineLoadOp`/`StoreOp`.
   *   5) Affinize selected operands of `memref.reinterpret_cast`,
   *      `tensor.extract_slice`, and `memref.subview`.
   *   6) Rebuild `scf::ForOp` with index-typed loop IV/bounds/step.
   *   7) Apply iterative DCE and remove redundant `arith.index_cast` ops.
   *
   * The pass signals failure if any unsupported signed division patterns were
   * encountered during expression building.
   */
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    bool hadError = false;
    module.walk([&](func::FuncOp func) {
      // Optionally, tag the last six args as symbols/dims when building maps.
      // We simply allow them as dims via the builder since they are function
      // block arguments.

      OpBuilder b(ctx);

      // Step 0: Convert all i32 function arguments to index type to eliminate
      // index_cast in the body and enable direct use in affine.apply.
      {
        FunctionType fty = func.getFunctionType();
        SmallVector<Type, 8> inputs(fty.getInputs().begin(),
                                    fty.getInputs().end());
        SmallVector<Type, 8> results(fty.getResults().begin(),
                                     fty.getResults().end());
        bool changed = false;
        for (unsigned i = 0, e = inputs.size(); i < e; ++i) {
          if (auto intTy = dyn_cast<IntegerType>(inputs[i])) {
            if (intTy.getWidth() == 32) {
              inputs[i] = IndexType::get(ctx);
              changed = true;
            }
          }
        }
        if (changed) {
          auto newTy = FunctionType::get(ctx, inputs, results);
          func.setType(newTy);
          // Update entry block argument types to match.
          Block &entry = func.getBody().front();
          for (unsigned i = 0, e = inputs.size(); i < e; ++i)
            entry.getArgument(i).setType(inputs[i]);

          // Canonicalize redundant index_cast (same src/dst type) after type
          // change.
          bool removed;
          int castIterGuard = 64;
          do {
            removed = false;
            SmallVector<Operation *, 32> casts;
            func.walk([&](arith::IndexCastOp ic) {
              if (ic.getIn().getType() == ic.getType())
                casts.push_back(ic.getOperation());
            });
            for (Operation *opCast : casts) {
              auto ic = cast<arith::IndexCastOp>(opCast);
              ic.getResult().replaceAllUsesWith(ic.getIn());
              ic.erase();
              removed = true;
            }
          } while (removed && --castIterGuard > 0);

          // Ensure integer arithmetic directly using grid args operates on
          // index type (promote constants and operands to index, rebuild ops).
          auto ensureIndex2 = [&](OpBuilder &rb, Operation *anchor,
                                  Value v) -> Value {
            if (v.getType().isIndex())
              return v;
            if (auto cint = v.getDefiningOp<arith::ConstantIntOp>()) {
              rb.setInsertionPoint(anchor);
              return rb.create<arith::ConstantIndexOp>(cint.getLoc(),
                                                       cint.value());
            }
            rb.setInsertionPoint(anchor);
            return rb.create<arith::IndexCastOp>(v.getLoc(),
                                                 IndexType::get(ctx), v);
          };

          bool changedArith;
          do {
            changedArith = false;
            SmallVector<Operation *, 64> arithToFix;
            func.walk([&](Operation *o) {
              if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp,
                      arith::DivSIOp, arith::MinSIOp, arith::MaxSIOp>(o)) {
                Type t0 = o->getOperand(0).getType();
                Type t1 = o->getOperand(1).getType();
                Type tr = o->getResult(0).getType();
                if (t0 != t1 || t0 != tr || !t0.isIndex() || !t1.isIndex() ||
                    !tr.isIndex())
                  arithToFix.push_back(o);
              }
            });
            for (Operation *o : arithToFix) {
              OpBuilder rb(o);
              Value a = o->getOperand(0);
              Value b = o->getOperand(1);
              Value ai = ensureIndex2(rb, o, a);
              Value bi = ensureIndex2(rb, o, b);
              Value newVal;
              if (auto addi = dyn_cast<arith::AddIOp>(o))
                newVal = rb.create<arith::AddIOp>(addi.getLoc(), ai, bi);
              else if (auto subi = dyn_cast<arith::SubIOp>(o))
                newVal = rb.create<arith::SubIOp>(subi.getLoc(), ai, bi);
              else if (auto muli = dyn_cast<arith::MulIOp>(o))
                newVal = rb.create<arith::MulIOp>(muli.getLoc(), ai, bi);
              else if (auto divi = dyn_cast<arith::DivSIOp>(o))
                newVal = rb.create<arith::DivSIOp>(divi.getLoc(), ai, bi);
              else if (auto mini = dyn_cast<arith::MinSIOp>(o))
                newVal = rb.create<arith::MinSIOp>(mini.getLoc(), ai, bi);
              else if (auto maxi = dyn_cast<arith::MaxSIOp>(o))
                newVal = rb.create<arith::MaxSIOp>(maxi.getLoc(), ai, bi);
              if (!newVal)
                continue;
              o->getResult(0).replaceAllUsesWith(newVal);
              o->erase();
              changedArith = true;
            }
          } while (changedArith);
        }
      }

      // Use exactly the last three function arguments as dims.
      SmallVector<Value, 3> spatialDims;
      if (func.getNumArguments() >= 3) {
        unsigned n = func.getNumArguments();
        for (unsigned i = n - 3; i < n; ++i)
          spatialDims.push_back(func.getArgument(i));
      }

      // Helper to try replacing an index value with affine.apply. Dims are the
      // last three function args plus surrounding loop IVs in the given scope.
      auto ensureAffine = [&](Value idx, Location loc,
                              Operation *scope) -> Value {
        if (!idx.getType().isIndex())
          return idx;
        // Trivially affine values: block arguments and constant indices.
        if (isa<BlockArgument>(idx) ||
            idx.getDefiningOp<arith::ConstantIndexOp>())
          return idx;
        // Compose dims: grid IDs (last 3 func args) + surrounding loop IVs.
        SmallVector<Value, 8> dimsAll(spatialDims.begin(), spatialDims.end());
        for (Operation *par = scope; par; par = par->getParentOp()) {
          if (auto loop = dyn_cast<scf::ForOp>(par))
            dimsAll.push_back(loop.getInductionVar());
          if (isa<func::FuncOp>(par))
            break;
        }
        LinearFormBuilder lfb(ctx, dimsAll, &hadError);
        AffineExpr expr = lfb.build(idx);
        if (!expr) {
          // Try partial conversion for add/sub when one side is affine.
          if (Operation *def = idx.getDefiningOp()) {
            if (auto addi = dyn_cast<arith::AddIOp>(def)) {
              // Try LHS
              if (AffineExpr el = lfb.build(addi.getLhs())) {
                SmallVector<Value, 8> od, os;
                AffineExpr elR = remapDimsAndSymbolsSequential(
                    el, od, lfb.getDims(), os, lfb.getSymbols());
                AffineMap mapL = AffineMap::get(od.size(), os.size(), elR);
                SmallVector<Value, 16> opsL;
                opsL.append(od.begin(), od.end());
                opsL.append(os.begin(), os.end());
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointAfter(def);
                Value lhsAff = b.create<affine::AffineApplyOp>(loc, mapL, opsL);
                return b.create<arith::AddIOp>(loc, lhsAff, addi.getRhs());
              }
              // Try RHS
              if (AffineExpr er = lfb.build(addi.getRhs())) {
                SmallVector<Value, 8> od, os;
                AffineExpr erR = remapDimsAndSymbolsSequential(
                    er, od, lfb.getDims(), os, lfb.getSymbols());
                AffineMap mapR = AffineMap::get(od.size(), os.size(), erR);
                SmallVector<Value, 16> opsR;
                opsR.append(od.begin(), od.end());
                opsR.append(os.begin(), os.end());
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointAfter(def);
                Value rhsAff = b.create<affine::AffineApplyOp>(loc, mapR, opsR);
                return b.create<arith::AddIOp>(loc, addi.getLhs(), rhsAff);
              }
            } else if (auto subi = dyn_cast<arith::SubIOp>(def)) {
              if (AffineExpr el = lfb.build(subi.getLhs())) {
                SmallVector<Value, 8> od, os;
                AffineExpr elR = remapDimsAndSymbolsSequential(
                    el, od, lfb.getDims(), os, lfb.getSymbols());
                AffineMap mapL = AffineMap::get(od.size(), os.size(), elR);
                SmallVector<Value, 16> opsL;
                opsL.append(od.begin(), od.end());
                opsL.append(os.begin(), os.end());
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointAfter(def);
                Value lhsAff = b.create<affine::AffineApplyOp>(loc, mapL, opsL);
                return b.create<arith::SubIOp>(loc, lhsAff, subi.getRhs());
              }
              if (AffineExpr er = lfb.build(subi.getRhs())) {
                SmallVector<Value, 8> od, os;
                AffineExpr erR = remapDimsAndSymbolsSequential(
                    er, od, lfb.getDims(), os, lfb.getSymbols());
                AffineMap mapR = AffineMap::get(od.size(), os.size(), erR);
                SmallVector<Value, 16> opsR;
                opsR.append(od.begin(), od.end());
                opsR.append(os.begin(), os.end());
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointAfter(def);
                Value rhsAff = b.create<affine::AffineApplyOp>(loc, mapR, opsR);
                return b.create<arith::SubIOp>(loc, subi.getLhs(), rhsAff);
              }
            }
          }
          return idx;
        }
        SmallVector<Value, 8> orderedDims, orderedSyms;
        AffineExpr remapped = remapDimsAndSymbolsSequential(
            expr, orderedDims, lfb.getDims(), orderedSyms, lfb.getSymbols());
        AffineMap map =
            AffineMap::get(orderedDims.size(), orderedSyms.size(), remapped);
        SmallVector<Value, 16> operands;
        operands.append(orderedDims.begin(), orderedDims.end());
        operands.append(orderedSyms.begin(), orderedSyms.end());
        if (Operation *def = idx.getDefiningOp())
          b.setInsertionPointAfter(def);
        else
          b.setInsertionPointToStart(&func.front());
        Value applied = b.create<affine::AffineApplyOp>(loc, map, operands);
        return applied;
      };

      // Walk loads/stores and attempt conversion to affine ops.
      func.walk([&](Operation *op) {
        // Convert scf.for to index-typed IV when possible by rebuilding the
        // loop with index-typed bounds/step. This eliminates i32 IV and the
        // associated index_casts inside the loop body.
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
          auto toIndex = [&](Value v) -> Value {
            if (v.getType().isIndex())
              return v;
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPoint(loop);
            if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>())
              return b.create<arith::ConstantIndexOp>(cInt.getLoc(),
                                                      cInt.value());
            return b.create<arith::IndexCastOp>(v.getLoc(), IndexType::get(ctx),
                                                v);
          };
          bool needIndex = !loop.getInductionVar().getType().isIndex() ||
                           !loop.getLowerBound().getType().isIndex() ||
                           !loop.getUpperBound().getType().isIndex() ||
                           !loop.getStep().getType().isIndex();
          bool needIterArgIndex = false;
          for (Value iterArg : loop.getRegionIterArgs()) {
            Type t = iterArg.getType();
            if (!t.isIndex() && llvm::isa<IntegerType>(t)) {
              needIterArgIndex = true;
              break;
            }
          }
          if (!needIndex && !needIterArgIndex)
            return;
          Value lbIdx = toIndex(loop.getLowerBound());
          Value ubIdx = toIndex(loop.getUpperBound());
          Value stIdx = toIndex(loop.getStep());
          // Affinize bounds and step where possible.
          Value lbAff = ensureAffine(lbIdx, loop.getLoc(), loop);
          Value ubAff = ensureAffine(ubIdx, loop.getLoc(), loop);
          Value stAff = ensureAffine(stIdx, loop.getLoc(), loop);
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPoint(loop);
          SmallVector<Value, 8> newInitArgs;
          newInitArgs.reserve(loop.getInitArgs().size());
          for (Value initV : loop.getInitArgs()) {
            Type t = initV.getType();
            if (!t.isIndex() && llvm::isa<IntegerType>(t))
              newInitArgs.push_back(toIndex(initV));
            else
              newInitArgs.push_back(initV);
          }
          auto newLoop = b.create<scf::ForOp>(loop.getLoc(), lbAff, ubAff,
                                              stAff, newInitArgs);
          // Clone body with IV remapped to index-typed IV.
          IRMapping mapper;
          Block &oldBody = *loop.getBody();
          Block &newBody = *newLoop.getBody();
          mapper.map(oldBody.getArgument(0), newBody.getArgument(0));
          for (auto it : llvm::enumerate(loop.getRegionIterArgs()))
            mapper.map(oldBody.getArgument(it.index() + 1),
                       newBody.getArgument(it.index() + 1));
          OpBuilder nb(&newBody, newBody.begin());
          for (Operation &inner : oldBody.without_terminator())
            nb.clone(inner, mapper);
          auto oldYield = cast<scf::YieldOp>(oldBody.getTerminator());
          SmallVector<Value, 8> yieldVals;
          yieldVals.reserve(oldYield.getNumOperands());
          for (Value v : oldYield.getOperands())
            yieldVals.push_back(mapper.lookupOrDefault(v));
          // Ensure yielded values match the new loop's iter_arg types. Our
          // earlier canonicalization may have turned integer math results into
          // index-typed values; cast them back to the expected types here.
          for (auto en : llvm::enumerate(yieldVals)) {
            unsigned i = en.index();
            Value yv = en.value();
            Type expectedTy = newLoop.getRegionIterArgs()[i].getType();
            Type actualTy = yv.getType();
            if (expectedTy == actualTy)
              continue;
            bool expectedIsIdx = expectedTy.isIndex();
            bool actualIsIdx = actualTy.isIndex();
            if ((expectedIsIdx && llvm::isa<IntegerType>(actualTy)) ||
                (actualIsIdx && llvm::isa<IntegerType>(expectedTy))) {
              yv = nb.create<arith::IndexCastOp>(oldYield.getLoc(), expectedTy,
                                                 yv);
              yieldVals[i] = yv;
            }
          }
          nb.create<scf::YieldOp>(oldYield.getLoc(), yieldVals);
          // Replace uses of the old loop results with the new ones, inserting
          // casts when the result types changed (e.g., i64 -> index).
          for (auto en : llvm::enumerate(loop.getResults())) {
            unsigned i = en.index();
            Value oldRes = en.value();
            Value newRes = newLoop.getResult(i);
            if (oldRes.getType() == newRes.getType()) {
              oldRes.replaceAllUsesWith(newRes);
              continue;
            }
            OpBuilder::InsertionGuard g2(b);
            b.setInsertionPointAfter(newLoop);
            // Only integer/index differences are expected here.
            Value casted = b.create<arith::IndexCastOp>(
                newLoop.getLoc(), oldRes.getType(), newRes);
            oldRes.replaceAllUsesWith(casted);
          }
          loop.erase();
          return;
        }
        // Rebuild reinterpret_cast with affine-applied offsets/sizes/strides
        // where possible.
        if (auto rc = dyn_cast<memref::ReinterpretCastOp>(op)) {
          // Compose dims: grid IDs (last 3 func args) + surrounding loop IVs.
          SmallVector<Value, 8> dimsAll(spatialDims.begin(), spatialDims.end());
          for (Operation *par = rc->getParentOp(); par;
               par = par->getParentOp()) {
            if (auto loop = dyn_cast<scf::ForOp>(par))
              dimsAll.push_back(loop.getInductionVar());
            if (isa<func::FuncOp>(par))
              break;
          }
          LinearFormBuilder lfb(ctx, dimsAll, &hadError);
          auto castToIndexAt = [&](Value v) -> Value {
            if (v.getType().isIndex())
              return v;
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPoint(rc);
            return b
                .create<arith::IndexCastOp>(rc.getLoc(), IndexType::get(ctx), v)
                .getResult();
          };
          SmallVector<Value, 4> newOffsets;
          newOffsets.reserve(rc.getOffsets().size());
          for (Value off : rc.getOffsets()) {
            AffineExpr e = lfb.build(off);
            if (e) {
              SmallVector<Value, 8> orderedDims, orderedSyms;
              AffineExpr remapped = remapDimsAndSymbolsSequential(
                  e, orderedDims, lfb.getDims(), orderedSyms, lfb.getSymbols());
              SmallVector<Value, 16> operands;
              for (Value d : orderedDims)
                operands.push_back(castToIndexAt(d));
              for (Value s : orderedSyms)
                operands.push_back(castToIndexAt(s));
              AffineMap m = AffineMap::get(orderedDims.size(),
                                           orderedSyms.size(), remapped);
              OpBuilder::InsertionGuard g(b);
              b.setInsertionPoint(rc);
              Value applied =
                  b.create<affine::AffineApplyOp>(rc.getLoc(), m, operands);
              newOffsets.push_back(applied);
            } else {
              newOffsets.push_back(off);
            }
          }
          // Keep sizes/strides unchanged to avoid introducing new dominance
          // issues; focus on offsets only for now.

          OpBuilder::InsertionGuard g(b);
          b.setInsertionPoint(rc);
          auto rebuilt = b.create<memref::ReinterpretCastOp>(
              rc.getLoc(), rc.getResult().getType(), rc.getSource(),
              ValueRange(newOffsets), rc.getSizes(), rc.getStrides(),
              rc.getStaticOffsetsAttr(), rc.getStaticSizesAttr(),
              rc.getStaticStridesAttr());
          rc.getResult().replaceAllUsesWith(rebuilt.getResult());
          rc.erase();
          return;
        }
        if (auto load = dyn_cast<memref::LoadOp>(op)) {
          SmallVector<Value, 4> newIdx;
          bool allOk = true;
          for (Value idx : load.getIndices()) {
            Value aff = ensureAffine(idx, load.getLoc(), load);
            if (aff == idx) {
              // Not necessarily a failure; idx might already be loop IV or
              // const. Accept if it is BlockArgument or ConstantIndex.
              if (!isa<BlockArgument>(idx) &&
                  !idx.getDefiningOp<arith::ConstantIndexOp>())
                allOk = false;
            }
            newIdx.push_back(aff);
          }
          if (!allOk)
            return;
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPoint(load);
          auto anew = b.create<affine::AffineLoadOp>(load.getLoc(),
                                                     load.getMemRef(), newIdx);
          load.getResult().replaceAllUsesWith(anew.getResult());
          load.erase();
          return;
        }
        if (auto store = dyn_cast<memref::StoreOp>(op)) {
          SmallVector<Value, 4> newIdx;
          bool allOk = true;
          for (Value idx : store.getIndices()) {
            Value aff = ensureAffine(idx, store.getLoc(), store);
            if (aff == idx) {
              if (!isa<BlockArgument>(idx) &&
                  !idx.getDefiningOp<arith::ConstantIndexOp>())
                allOk = false;
            }
            newIdx.push_back(aff);
          }
          if (!allOk)
            return;
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPoint(store);
          b.create<affine::AffineStoreOp>(store.getLoc(),
                                          store.getValueToStore(),
                                          store.getMemRef(), newIdx);
          store.erase();
          return;
        }

        // Try to affinize sizes of extract_slice (tensor) and subview (memref).
        if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op)) {
          // Compose dims: grid IDs (last 3 func args) + surrounding loop IVs.
          SmallVector<Value, 8> dimsAll(spatialDims.begin(), spatialDims.end());
          for (Operation *par = slice->getParentOp(); par;
               par = par->getParentOp()) {
            if (auto loop = dyn_cast<scf::ForOp>(par))
              dimsAll.push_back(loop.getInductionVar());
            if (isa<func::FuncOp>(par))
              break;
          }
          LinearFormBuilder lfb(ctx, dimsAll, &hadError);
          auto castToIndexAt = [&](Value v) -> Value {
            if (v.getType().isIndex())
              return v;
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPoint(slice);
            return b
                .create<arith::IndexCastOp>(slice.getLoc(), IndexType::get(ctx),
                                            v)
                .getResult();
          };

          SmallVector<Value, 4> newSizes;
          newSizes.reserve(slice.getSizes().size());
          auto convertValue = [&](Value val) -> Value {
            auto unwrapIndexCast = [](Value v) -> Value {
              if (auto ic =
                      dyn_cast_or_null<arith::IndexCastOp>(v.getDefiningOp()))
                return ic.getIn();
              return v;
            };
            auto isConstIdx = [&](Value v, int64_t &cst) -> bool {
              if (auto c = dyn_cast_or_null<arith::ConstantIndexOp>(
                      v.getDefiningOp())) {
                cst = c.value();
                return true;
              }
              return false;
            };
            auto tryClampTilePattern = [&](Value v) -> Value {
              // Match: min( [optional min(..., T)], sub(max(min(base+T, dim),
              // base), base), T)
              arith::MinSIOp minTop =
                  dyn_cast_or_null<arith::MinSIOp>(v.getDefiningOp());
              if (!minTop)
                return Value();
              Value lhs = minTop.getLhs();
              Value rhs = minTop.getRhs();
              int64_t tileC = -1;
              Value tileVal;
              if (isConstIdx(rhs, tileC)) {
                tileVal = rhs;
              } else if (isConstIdx(lhs, tileC)) {
                tileVal = lhs;
                lhs = rhs;
              } else {
                return Value();
              }

              // Collapse a redundant inner min(..., T)
              if (auto innerMin =
                      dyn_cast_or_null<arith::MinSIOp>(lhs.getDefiningOp())) {
                int64_t tile2;
                if (isConstIdx(innerMin.getLhs(), tile2) ||
                    isConstIdx(innerMin.getRhs(), tile2)) {
                  if (tile2 == tileC)
                    lhs = (innerMin.getLhs() == tileVal) ? innerMin.getRhs()
                                                         : innerMin.getLhs();
                }
              }

              auto sub = dyn_cast_or_null<arith::SubIOp>(lhs.getDefiningOp());
              if (!sub)
                return Value();
              Value y = sub.getLhs();
              Value base = sub.getRhs();

              auto max = dyn_cast_or_null<arith::MaxSIOp>(y.getDefiningOp());
              if (!max)
                return Value();
              Value mb0 = max.getLhs();
              Value mb1 = max.getRhs();
              Value w;
              if (mb0 == base)
                w = mb1;
              else if (mb1 == base)
                w = mb0;
              else
                return Value();

              auto min2 = dyn_cast_or_null<arith::MinSIOp>(w.getDefiningOp());
              if (!min2)
                return Value();
              Value a = min2.getLhs();
              Value rhsVal = min2.getRhs();
              Value add;
              Value dim;
              // Identify add(base, T) vs dim (possibly with index_cast)
              auto isAddBaseTile = [&](Value v) -> bool {
                auto addi = dyn_cast_or_null<arith::AddIOp>(v.getDefiningOp());
                if (!addi)
                  return false;
                // Accept either operand equal to base and the other constant T.
                int64_t c;
                if (addi.getLhs() == base && isConstIdx(addi.getRhs(), c) &&
                    c == tileC)
                  return true;
                if (addi.getRhs() == base && isConstIdx(addi.getLhs(), c) &&
                    c == tileC)
                  return true;
                return false;
              };
              if (isAddBaseTile(a)) {
                add = a;
                dim = unwrapIndexCast(rhsVal);
              } else if (isAddBaseTile(rhsVal)) {
                add = rhsVal;
                dim = unwrapIndexCast(a);
              } else {
                return Value();
              }

              // We matched the clamp pattern. Build affine.min with {dim -
              // base, T}.
              AffineExpr eDim = lfb.build(dim);
              AffineExpr eBase = lfb.build(base);
              if (!eDim || !eBase)
                return Value();
              AffineExpr eDiff = eDim - eBase;
              SmallVector<Value, 8> orderedSyms;
              AffineExpr eRem =
                  remapSymbolsSequential(eDiff, orderedSyms, lfb.getSymbols());
              SmallVector<Value, 16> operands;
              for (Value d : lfb.getDims())
                operands.push_back(castToIndexAt(d));
              for (Value s : orderedSyms)
                operands.push_back(castToIndexAt(s));
              AffineMap map = AffineMap::get(
                  static_cast<unsigned>(lfb.getDims().size()),
                  static_cast<unsigned>(orderedSyms.size()),
                  ArrayRef<AffineExpr>{eRem, getAffineConstantExpr(tileC, ctx)},
                  ctx);
              OpBuilder::InsertionGuard g(b);
              b.setInsertionPoint(slice);
              return b
                  .create<affine::AffineMinOp>(slice.getLoc(), map, operands)
                  .getResult();
            };

            if (Value v = tryClampTilePattern(val))
              return v;
            // First try a pure affine.apply expression with compact dims/syms.
            if (AffineExpr e = lfb.build(val)) {
              SmallVector<Value, 8> orderedDims, orderedSyms;
              AffineExpr remapped = remapDimsAndSymbolsSequential(
                  e, orderedDims, lfb.getDims(), orderedSyms, lfb.getSymbols());
              SmallVector<Value, 16> operands;
              for (Value d : orderedDims)
                operands.push_back(castToIndexAt(d));
              for (Value s : orderedSyms)
                operands.push_back(castToIndexAt(s));
              AffineMap map = AffineMap::get(orderedDims.size(),
                                             orderedSyms.size(), remapped);
              OpBuilder::InsertionGuard g(b);
              b.setInsertionPoint(slice);
              return b.create<affine::AffineApplyOp>(slice.getLoc(), map,
                                                     operands);
            }
            // Handle min/max of affine expressions via affine.min/max ops.
            if (auto minsi =
                    dyn_cast_or_null<arith::MinSIOp>(val.getDefiningOp())) {
              AffineExpr el = lfb.build(minsi.getLhs());
              AffineExpr er = lfb.build(minsi.getRhs());
              if (el && er) {
                // Unify dims and symbols across both expressions.
                SmallVector<unsigned, 8> useSymL, useSymR, useDimL, useDimR;
                collectUsedSymbolPositions(el, useSymL);
                collectUsedSymbolPositions(er, useSymR);
                collectUsedDimPositions(el, useDimL);
                collectUsedDimPositions(er, useDimR);
                SmallVector<unsigned, 16> dimOrder;
                dimOrder.append(useDimL.begin(), useDimL.end());
                for (unsigned p : useDimR)
                  if (llvm::find(dimOrder, p) == dimOrder.end())
                    dimOrder.push_back(p);
                DenseMap<unsigned, unsigned> oldDimToNew;
                for (unsigned i = 0, eN = dimOrder.size(); i < eN; ++i)
                  oldDimToNew.try_emplace(dimOrder[i], i);
                AffineExpr elDimR = remapDimsWithMapping(el, oldDimToNew);
                AffineExpr erDimR = remapDimsWithMapping(er, oldDimToNew);
                SmallVector<unsigned, 16> symOrder;
                symOrder.append(useSymL.begin(), useSymL.end());
                for (unsigned p : useSymR)
                  if (llvm::find(symOrder, p) == symOrder.end())
                    symOrder.push_back(p);
                DenseMap<unsigned, unsigned> oldSymToNew;
                for (unsigned i = 0, eN = symOrder.size(); i < eN; ++i)
                  oldSymToNew.try_emplace(symOrder[i], i);
                AffineExpr elR = remapSymbolsWithMapping(elDimR, oldSymToNew);
                AffineExpr erR = remapSymbolsWithMapping(erDimR, oldSymToNew);
                SmallVector<Value, 16> operands;
                SmallVector<Value, 8> orderedDims;
                orderedDims.reserve(dimOrder.size());
                for (unsigned pos : dimOrder)
                  orderedDims.push_back(castToIndexAt(lfb.getDims()[pos]));
                for (Value d : orderedDims)
                  operands.push_back(d);
                SmallVector<Value, 8> orderedSyms;
                orderedSyms.reserve(symOrder.size());
                for (unsigned pos : symOrder)
                  orderedSyms.push_back(castToIndexAt(lfb.getSymbols()[pos]));
                operands.append(orderedSyms.begin(), orderedSyms.end());
                AffineMap mCombined =
                    AffineMap::get(static_cast<unsigned>(orderedDims.size()),
                                   static_cast<unsigned>(orderedSyms.size()),
                                   ArrayRef<AffineExpr>{elR, erR}, ctx);
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPoint(slice);
                return b
                    .create<affine::AffineMinOp>(slice.getLoc(), mCombined,
                                                 operands)
                    .getResult();
              }
            }
            if (auto maxsi =
                    dyn_cast_or_null<arith::MaxSIOp>(val.getDefiningOp())) {
              AffineExpr el = lfb.build(maxsi.getLhs());
              AffineExpr er = lfb.build(maxsi.getRhs());
              if (el && er) {
                SmallVector<unsigned, 8> useSymL, useSymR, useDimL, useDimR;
                collectUsedSymbolPositions(el, useSymL);
                collectUsedSymbolPositions(er, useSymR);
                collectUsedDimPositions(el, useDimL);
                collectUsedDimPositions(er, useDimR);
                SmallVector<unsigned, 16> dimOrder;
                dimOrder.append(useDimL.begin(), useDimL.end());
                for (unsigned p : useDimR)
                  if (llvm::find(dimOrder, p) == dimOrder.end())
                    dimOrder.push_back(p);
                DenseMap<unsigned, unsigned> oldDimToNew;
                for (unsigned i = 0, eN = dimOrder.size(); i < eN; ++i)
                  oldDimToNew.try_emplace(dimOrder[i], i);
                AffineExpr elDimR = remapDimsWithMapping(el, oldDimToNew);
                AffineExpr erDimR = remapDimsWithMapping(er, oldDimToNew);
                SmallVector<unsigned, 16> symOrder;
                symOrder.append(useSymL.begin(), useSymL.end());
                for (unsigned p : useSymR)
                  if (llvm::find(symOrder, p) == symOrder.end())
                    symOrder.push_back(p);
                DenseMap<unsigned, unsigned> oldSymToNew;
                for (unsigned i = 0, eN = symOrder.size(); i < eN; ++i)
                  oldSymToNew.try_emplace(symOrder[i], i);
                AffineExpr elR = remapSymbolsWithMapping(elDimR, oldSymToNew);
                AffineExpr erR = remapSymbolsWithMapping(erDimR, oldSymToNew);
                SmallVector<Value, 16> operands;
                SmallVector<Value, 8> orderedDims;
                orderedDims.reserve(dimOrder.size());
                for (unsigned pos : dimOrder)
                  orderedDims.push_back(castToIndexAt(lfb.getDims()[pos]));
                for (Value d : orderedDims)
                  operands.push_back(d);
                SmallVector<Value, 8> orderedSyms;
                orderedSyms.reserve(symOrder.size());
                for (unsigned pos : symOrder)
                  orderedSyms.push_back(castToIndexAt(lfb.getSymbols()[pos]));
                operands.append(orderedSyms.begin(), orderedSyms.end());
                AffineMap mCombined =
                    AffineMap::get(static_cast<unsigned>(orderedDims.size()),
                                   static_cast<unsigned>(orderedSyms.size()),
                                   ArrayRef<AffineExpr>{elR, erR}, ctx);
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPoint(slice);
                return b
                    .create<affine::AffineMaxOp>(slice.getLoc(), mCombined,
                                                 operands)
                    .getResult();
              }
            }
            return val;
          };

          for (Value sz : slice.getSizes())
            newSizes.push_back(convertValue(sz));

          // Rebuild the op only if any size changed to an affine op or apply.
          bool changed =
              llvm::any_of(llvm::zip(slice.getSizes(), newSizes), [](auto it) {
                return std::get<0>(it) != std::get<1>(it);
              });
          if (!changed)
            return;

          OpBuilder::InsertionGuard g(b);
          b.setInsertionPoint(slice);
          auto rebuilt = b.create<tensor::ExtractSliceOp>(
              slice.getLoc(), slice.getType(), slice.getSource(),
              slice.getOffsets(), ValueRange(newSizes), slice.getStrides(),
              slice.getStaticOffsets(), slice.getStaticSizes(),
              slice.getStaticStrides());
          slice.getResult().replaceAllUsesWith(rebuilt.getResult());
          slice.erase();
          return;
        }
        if (auto subv = dyn_cast<memref::SubViewOp>(op)) {
          SmallVector<Value, 8> dimsAll(spatialDims.begin(), spatialDims.end());
          for (Operation *par = subv->getParentOp(); par;
               par = par->getParentOp()) {
            if (auto loop = dyn_cast<scf::ForOp>(par))
              dimsAll.push_back(loop.getInductionVar());
            if (isa<func::FuncOp>(par))
              break;
          }
          LinearFormBuilder lfb(ctx, dimsAll, &hadError);
          auto castToIndexAt = [&](Value v) -> Value {
            if (v.getType().isIndex())
              return v;
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPoint(subv);
            return b
                .create<arith::IndexCastOp>(subv.getLoc(), IndexType::get(ctx),
                                            v)
                .getResult();
          };

          SmallVector<Value, 4> newSizes;
          newSizes.reserve(subv.getSizes().size());
          auto convertValue = [&](Value val) -> Value {
            auto unwrapIndexCast = [](Value v) -> Value {
              if (auto ic =
                      dyn_cast_or_null<arith::IndexCastOp>(v.getDefiningOp()))
                return ic.getIn();
              return v;
            };
            auto isConstIdx = [&](Value v, int64_t &cst) -> bool {
              if (auto c = dyn_cast_or_null<arith::ConstantIndexOp>(
                      v.getDefiningOp())) {
                cst = c.value();
                return true;
              }
              return false;
            };
            auto tryClampTilePattern = [&](Value v) -> Value {
              arith::MinSIOp minTop =
                  dyn_cast_or_null<arith::MinSIOp>(v.getDefiningOp());
              if (!minTop)
                return Value();
              Value lhs = minTop.getLhs();
              Value rhs = minTop.getRhs();
              int64_t tileC = -1;
              Value tileVal;
              if (isConstIdx(rhs, tileC)) {
                tileVal = rhs;
              } else if (isConstIdx(lhs, tileC)) {
                tileVal = lhs;
                lhs = rhs;
              } else {
                return Value();
              }
              if (auto innerMin =
                      dyn_cast_or_null<arith::MinSIOp>(lhs.getDefiningOp())) {
                int64_t tile2;
                if (isConstIdx(innerMin.getLhs(), tile2) ||
                    isConstIdx(innerMin.getRhs(), tile2)) {
                  if (tile2 == tileC)
                    lhs = (innerMin.getLhs() == tileVal) ? innerMin.getRhs()
                                                         : innerMin.getLhs();
                }
              }
              auto sub = dyn_cast_or_null<arith::SubIOp>(lhs.getDefiningOp());
              if (!sub)
                return Value();
              Value y = sub.getLhs();
              Value base = sub.getRhs();
              auto max = dyn_cast_or_null<arith::MaxSIOp>(y.getDefiningOp());
              if (!max)
                return Value();
              Value mb0 = max.getLhs();
              Value mb1 = max.getRhs();
              Value w;
              if (mb0 == base)
                w = mb1;
              else if (mb1 == base)
                w = mb0;
              else
                return Value();
              auto min2 = dyn_cast_or_null<arith::MinSIOp>(w.getDefiningOp());
              if (!min2)
                return Value();
              Value a = min2.getLhs();
              Value rhsVal = min2.getRhs();
              auto isAddBaseTile = [&](Value v) -> bool {
                auto addi = dyn_cast_or_null<arith::AddIOp>(v.getDefiningOp());
                if (!addi)
                  return false;
                int64_t c;
                if (addi.getLhs() == base && isConstIdx(addi.getRhs(), c) &&
                    c == tileC)
                  return true;
                if (addi.getRhs() == base && isConstIdx(addi.getLhs(), c) &&
                    c == tileC)
                  return true;
                return false;
              };
              Value add;
              Value dim;
              if (isAddBaseTile(a)) {
                add = a;
                dim = unwrapIndexCast(rhsVal);
              } else if (isAddBaseTile(rhsVal)) {
                add = rhsVal;
                dim = unwrapIndexCast(a);
              } else {
                return Value();
              }
              AffineExpr eDim = lfb.build(dim);
              AffineExpr eBase = lfb.build(base);
              if (!eDim || !eBase)
                return Value();
              AffineExpr eDiff = eDim - eBase;
              SmallVector<Value, 8> orderedSyms;
              AffineExpr eRem =
                  remapSymbolsSequential(eDiff, orderedSyms, lfb.getSymbols());
              SmallVector<Value, 16> operands;
              for (Value d : lfb.getDims())
                operands.push_back(castToIndexAt(d));
              for (Value s : orderedSyms)
                operands.push_back(castToIndexAt(s));
              AffineMap map = AffineMap::get(
                  static_cast<unsigned>(lfb.getDims().size()),
                  static_cast<unsigned>(orderedSyms.size()),
                  ArrayRef<AffineExpr>{eRem, getAffineConstantExpr(tileC, ctx)},
                  ctx);
              OpBuilder::InsertionGuard g(b);
              b.setInsertionPoint(subv);
              return b.create<affine::AffineMinOp>(subv.getLoc(), map, operands)
                  .getResult();
            };

            if (Value v = tryClampTilePattern(val))
              return v;
            if (AffineExpr e = lfb.build(val)) {
              SmallVector<Value, 8> orderedDims, orderedSyms;
              AffineExpr remapped = remapDimsAndSymbolsSequential(
                  e, orderedDims, lfb.getDims(), orderedSyms, lfb.getSymbols());
              SmallVector<Value, 16> operands;
              for (Value d : orderedDims)
                operands.push_back(castToIndexAt(d));
              for (Value s : orderedSyms)
                operands.push_back(castToIndexAt(s));
              AffineMap map = AffineMap::get(orderedDims.size(),
                                             orderedSyms.size(), remapped);
              OpBuilder::InsertionGuard g(b);
              b.setInsertionPoint(subv);
              return b.create<affine::AffineApplyOp>(subv.getLoc(), map,
                                                     operands);
            }
            if (auto minsi =
                    dyn_cast_or_null<arith::MinSIOp>(val.getDefiningOp())) {
              AffineExpr el = lfb.build(minsi.getLhs());
              AffineExpr er = lfb.build(minsi.getRhs());
              if (el && er) {
                SmallVector<unsigned, 8> useSymL, useSymR, useDimL, useDimR;
                collectUsedSymbolPositions(el, useSymL);
                collectUsedSymbolPositions(er, useSymR);
                collectUsedDimPositions(el, useDimL);
                collectUsedDimPositions(er, useDimR);
                SmallVector<unsigned, 16> dimOrder;
                dimOrder.append(useDimL.begin(), useDimL.end());
                for (unsigned p : useDimR)
                  if (llvm::find(dimOrder, p) == dimOrder.end())
                    dimOrder.push_back(p);
                DenseMap<unsigned, unsigned> oldDimToNew;
                for (unsigned i = 0, eN = dimOrder.size(); i < eN; ++i)
                  oldDimToNew.try_emplace(dimOrder[i], i);
                AffineExpr elDimR = remapDimsWithMapping(el, oldDimToNew);
                AffineExpr erDimR = remapDimsWithMapping(er, oldDimToNew);
                SmallVector<unsigned, 16> symOrder;
                symOrder.append(useSymL.begin(), useSymL.end());
                for (unsigned p : useSymR)
                  if (llvm::find(symOrder, p) == symOrder.end())
                    symOrder.push_back(p);
                DenseMap<unsigned, unsigned> oldSymToNew;
                for (unsigned i = 0, eN = symOrder.size(); i < eN; ++i)
                  oldSymToNew.try_emplace(symOrder[i], i);
                AffineExpr elR = remapSymbolsWithMapping(elDimR, oldSymToNew);
                AffineExpr erR = remapSymbolsWithMapping(erDimR, oldSymToNew);
                SmallVector<Value, 16> operands;
                SmallVector<Value, 8> orderedDims;
                orderedDims.reserve(dimOrder.size());
                for (unsigned pos : dimOrder)
                  orderedDims.push_back(castToIndexAt(lfb.getDims()[pos]));
                for (Value d : orderedDims)
                  operands.push_back(d);
                SmallVector<Value, 8> orderedSyms;
                orderedSyms.reserve(symOrder.size());
                for (unsigned pos : symOrder)
                  orderedSyms.push_back(castToIndexAt(lfb.getSymbols()[pos]));
                operands.append(orderedSyms.begin(), orderedSyms.end());
                AffineMap mCombined =
                    AffineMap::get(static_cast<unsigned>(orderedDims.size()),
                                   static_cast<unsigned>(orderedSyms.size()),
                                   ArrayRef<AffineExpr>{elR, erR}, ctx);
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPoint(subv);
                return b
                    .create<affine::AffineMinOp>(subv.getLoc(), mCombined,
                                                 operands)
                    .getResult();
              }
            }
            if (auto maxsi =
                    dyn_cast_or_null<arith::MaxSIOp>(val.getDefiningOp())) {
              AffineExpr el = lfb.build(maxsi.getLhs());
              AffineExpr er = lfb.build(maxsi.getRhs());
              if (el && er) {
                SmallVector<unsigned, 8> useL, useR;
                collectUsedSymbolPositions(el, useL);
                collectUsedSymbolPositions(er, useR);
                SmallVector<unsigned, 16> order;
                order.append(useL.begin(), useL.end());
                for (unsigned p : useR)
                  if (llvm::find(order, p) == order.end())
                    order.push_back(p);
                DenseMap<unsigned, unsigned> oldToNew;
                for (unsigned i = 0, e = order.size(); i < e; ++i)
                  oldToNew.try_emplace(order[i], i);
                AffineExpr elR = remapSymbolsWithMapping(el, oldToNew);
                AffineExpr erR = remapSymbolsWithMapping(er, oldToNew);
                SmallVector<Value, 16> operands;
                for (Value d : lfb.getDims())
                  operands.push_back(castToIndexAt(d));
                SmallVector<Value, 8> orderedSyms;
                orderedSyms.reserve(order.size());
                for (unsigned pos : order)
                  orderedSyms.push_back(castToIndexAt(lfb.getSymbols()[pos]));
                operands.append(orderedSyms.begin(), orderedSyms.end());
                AffineMap mCombined =
                    AffineMap::get(static_cast<unsigned>(lfb.getDims().size()),
                                   static_cast<unsigned>(orderedSyms.size()),
                                   ArrayRef<AffineExpr>{elR, erR}, ctx);
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPoint(subv);
                return b
                    .create<affine::AffineMaxOp>(subv.getLoc(), mCombined,
                                                 operands)
                    .getResult();
              }
            }
            return val;
          };

          for (Value sz : subv.getSizes())
            newSizes.push_back(convertValue(sz));

          bool changed =
              llvm::any_of(llvm::zip(subv.getSizes(), newSizes), [](auto it) {
                return std::get<0>(it) != std::get<1>(it);
              });
          if (!changed)
            return;

          OpBuilder::InsertionGuard g(b);
          b.setInsertionPoint(subv);
          auto rebuilt = b.create<memref::SubViewOp>(
              subv.getLoc(), subv.getType(), subv.getSource(),
              subv.getOffsets(), ValueRange(newSizes), subv.getStrides(),
              subv.getStaticOffsets(), subv.getStaticSizes(),
              subv.getStaticStrides());
          subv.getResult().replaceAllUsesWith(rebuilt.getResult());
          subv.erase();
          return;
        }
      });

      // General dead-code elimination: remove any trivially-dead operations.
      // Iterate to a fixed point because erasing may unlock more DCE.
      bool erased;
      int dceIterGuard = 1024;
      do {
        erased = false;
        SmallVector<Operation *, 64> toErase;
        func.walk([&](Operation *o) {
          if (isOpTriviallyDead(o))
            toErase.push_back(o);
        });
        if (!toErase.empty()) {
          for (Operation *o : toErase)
            o->erase();
          erased = true;
        }
      } while (erased && --dceIterGuard > 0);

      // Remove redundant index_cast (same src/dst types) that may remain.
      bool removedCasts;
      int finalCastIterGuard = 64;
      do {
        removedCasts = false;
        SmallVector<Operation *, 64> redCasts;
        func.walk([&](arith::IndexCastOp ic) {
          if (ic.getIn().getType() == ic.getType())
            redCasts.push_back(ic.getOperation());
        });
        for (Operation *opCast : redCasts) {
          auto ic = cast<arith::IndexCastOp>(opCast);
          ic.getResult().replaceAllUsesWith(ic.getIn());
          ic.erase();
          removedCasts = true;
        }
      } while (removedCasts && --finalCastIterGuard > 0);
    });
    if (hadError)
      signalPassFailure();
  }
};

} // namespace

/**
 * @brief Create an instance of the Triton-shared affinization pass.
 *
 * @return New pass instance that converts Triton-shared index arithmetic to
 *         affine IR and canonicalizes loads/stores and related ops.
 */
std::unique_ptr<mlir::Pass> createTritonSharedAffinizePass() {
  return std::make_unique<TritonSharedAffinizePass>();
}

/**
 * @brief Register the Triton-shared affinization pass with MLIR.
 *
 * @details After registration, the pass can be invoked by name
 * `--loom-triton-shared-affinize` in pass pipelines.
 */
void registerTritonSharedAffinizePass() {
  PassRegistration<TritonSharedAffinizePass>();
}

} // namespace passes
} // namespace loom
