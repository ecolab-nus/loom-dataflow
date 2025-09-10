//===- triton_shared_affinize.cpp - Affinize Triton-shared indices -------===//
//
// This pass converts arithmetic index expressions in Triton-shared lowered
// kernels into affine.apply ops and replaces eligible memory ops with their
// affine counterparts. The last six function arguments that encode the GPU
// grid/thread sizes and IDs are modeled as symbols/dims in affine maps.
//
// The transformation is conservative: only expressions that can be proven to be
// affine combinations of loop IVs, function arguments, and constants are
// converted.
//
//===----------------------------------------------------------------------===//

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
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace tmd {
namespace passes {

namespace {

/// Affine expression builder over an arith expression graph.
class AffineExprBuilder {
public:
  AffineExprBuilder(MLIRContext *ctx) : ctx(ctx) {}

  /// Try to build an AffineExpr for the given value, collecting its
  /// dims/symbols. Returns null AffineExpr on failure.
  AffineExpr buildFromIndexValue(Value v, SmallVectorImpl<Value> &dims,
                                 SmallVectorImpl<Value> &symbols) {
    return build(v, dims, symbols);
  }

private:
  AffineExpr build(Value v, SmallVectorImpl<Value> &dims,
                   SmallVectorImpl<Value> &symbols) {
    if (!v.getType().isIndex())
      return AffineExpr();

    // Direct mapping: loop IVs and function block arguments as dims.
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      Operation *owner = barg.getOwner()->getParentOp();
      if (isa_and_nonnull<scf::ForOp>(owner) ||
          isa_and_nonnull<func::FuncOp>(owner)) {
        return getOrAddDim(barg, dims);
      }
    }

    Operation *def = v.getDefiningOp();
    if (!def)
      return AffineExpr();

    // Propagate through index casts.
    if (auto ic = dyn_cast<arith::IndexCastOp>(def))
      return build(ic.getIn(), dims, symbols);

    OpBuilder b(ctx);
    if (auto cidx = dyn_cast<arith::ConstantIndexOp>(def))
      return getAffineConstantExpr(cidx.value(), ctx);

    // i32 constants frequently appear then cast to index; support those too.
    if (auto csi = dyn_cast<arith::ConstantIntOp>(def))
      return getAffineConstantExpr(csi.value(), ctx);

    // Handle simple affine-preserving arith ops.
    if (auto addi = dyn_cast<arith::AddIOp>(def)) {
      AffineExpr a = build(addi.getLhs(), dims, symbols);
      AffineExpr b = build(addi.getRhs(), dims, symbols);
      if (a && b)
        return a + b;
      return AffineExpr();
    }
    if (auto subi = dyn_cast<arith::SubIOp>(def)) {
      AffineExpr a = build(subi.getLhs(), dims, symbols);
      AffineExpr b = build(subi.getRhs(), dims, symbols);
      if (a && b)
        return a - b;
      return AffineExpr();
    }
    if (auto muli = dyn_cast<arith::MulIOp>(def)) {
      // Only allow multiply by constant.
      AffineExpr a = build(muli.getLhs(), dims, symbols);
      AffineExpr b = build(muli.getRhs(), dims, symbols);
      if (a && b) {
        // One of a/b must be a constant.
        if (auto ca = dyn_cast<AffineConstantExpr>(a))
          return b * ca.getValue();
        if (auto cb = dyn_cast<AffineConstantExpr>(b))
          return a * cb.getValue();
      }
      return AffineExpr();
    }
    if (auto divi = dyn_cast<arith::DivSIOp>(def)) {
      AffineExpr a = build(divi.getLhs(), dims, symbols);
      AffineExpr b = build(divi.getRhs(), dims, symbols);
      if (a && b)
        if (auto cb = dyn_cast<AffineConstantExpr>(b))
          if (cb.getValue() != 0)
            return a.floorDiv(cb.getValue());
      return AffineExpr();
    }

    // Default: unsupported op in affine context.
    return AffineExpr();
  }

  AffineExpr getOrAddDim(Value v, SmallVectorImpl<Value> &dims) {
    for (unsigned i = 0, e = dims.size(); i < e; ++i)
      if (dims[i] == v)
        return getAffineDimExpr(i, ctx);
    dims.push_back(v);
    return getAffineDimExpr(dims.size() - 1, ctx);
  }

  MLIRContext *ctx;
};

/// Builder that constructs an affine expression using fixed dims (the six
/// spatial arguments) and promotes any non-affine sub-expressions to symbols.
class LinearFormBuilder {
public:
  LinearFormBuilder(MLIRContext *ctx, ArrayRef<Value> fixedDims) : ctx(ctx) {
    dims.assign(fixedDims.begin(), fixedDims.end());
    for (unsigned i = 0; i < dims.size(); ++i)
      dimIndex.try_emplace(dims[i], i);
  }

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

    // Division by constant allowed (floorDiv), else promote to symbol.
    if (auto divi = dyn_cast_or_null<arith::DivSIOp>(v.getDefiningOp())) {
      AffineExpr a = build(divi.getLhs());
      AffineExpr b = build(divi.getRhs());
      if (auto cb = dyn_cast<AffineConstantExpr>(b))
        if (a && cb.getValue() != 0)
          return cache[v] = a.floorDiv(cb.getValue());
      return cache[v] = getOrAddSymbol(v);
    }

    // Block arguments not among fixed dims are symbols (including loop IVs).
    if (isa<BlockArgument>(v))
      return cache[v] = getOrAddSymbol(v);

    // Default: promote to symbol.
    return cache[v] = getOrAddSymbol(v);
  }

  ArrayRef<Value> getDims() const { return dims; }
  ArrayRef<Value> getSymbols() const { return symbols; }

private:
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
};

/// Collect symbol indices used by an affine expression.
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

/// Pass that performs the affinization.
class TritonSharedAffinizePass
    : public PassWrapper<TritonSharedAffinizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonSharedAffinizePass)

  StringRef getArgument() const override {
    return "tmd-triton-shared-affinize";
  }
  StringRef getDescription() const override {
    return "Convert Triton-shared index arithmetic to affine.apply and affine "
           "memops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect, func::FuncDialect, scf::SCFDialect,
                    linalg::LinalgDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    module.walk([&](func::FuncOp func) {
      // Optionally, tag the last six args as symbols/dims when building maps.
      // We simply allow them as dims via the builder since they are function
      // block arguments.

      OpBuilder b(ctx);
      AffineExprBuilder aeb(ctx);

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
          int arithIterGuard = 256;
          do {
            changedArith = false;
            SmallVector<Operation *, 64> arithToFix;
            func.walk([&](Operation *o) {
              if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp,
                      arith::DivSIOp, arith::MinSIOp, arith::MaxSIOp>(o)) {
                Type t0 = o->getOperand(0).getType();
                Type t1 = o->getOperand(1).getType();
                Type tr = o->getResult(0).getType();
                if (t0 != t1 || t0 != tr || t0.isIndex() || t1.isIndex() ||
                    tr.isIndex())
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
          } while (changedArith && --arithIterGuard > 0);
        }
      }

      // Helper to try replacing an index value with affine.apply
      auto ensureAffine = [&](Value idx, Location loc) -> Value {
        if (!idx.getType().isIndex())
          return idx;
        // Trivially affine values: block arguments and constant indices.
        if (isa<BlockArgument>(idx) ||
            idx.getDefiningOp<arith::ConstantIndexOp>())
          return idx;
        SmallVector<Value, 8> dims, symbols;
        AffineExpr expr = aeb.buildFromIndexValue(idx, dims, symbols);
        if (!expr) {
          // Try partial conversion for add/sub when one side is affine.
          if (Operation *def = idx.getDefiningOp()) {
            if (auto addi = dyn_cast<arith::AddIOp>(def)) {
              // Directly express as affine sum of two dims if both are
              // index-typed.
              if (addi.getLhs().getType().isIndex() &&
                  addi.getRhs().getType().isIndex()) {
                SmallVector<Value, 2> ds = {addi.getLhs(), addi.getRhs()};
                AffineMap m = AffineMap::get(/*dims=*/2, /*symbols=*/0,
                                             getAffineDimExpr(0, ctx) +
                                                 getAffineDimExpr(1, ctx));
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointAfter(def);
                return b.create<affine::AffineApplyOp>(loc, m, ds);
              }
              // Try LHS
              SmallVector<Value, 8> dimsL, symsL;
              if (AffineExpr el =
                      aeb.buildFromIndexValue(addi.getLhs(), dimsL, symsL)) {
                AffineMap mapL = AffineMap::get(dimsL.size(), symsL.size(), el);
                SmallVector<Value, 16> opsL;
                opsL.append(dimsL.begin(), dimsL.end());
                opsL.append(symsL.begin(), symsL.end());
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointAfter(def);
                Value lhsAff = b.create<affine::AffineApplyOp>(loc, mapL, opsL);
                return b.create<arith::AddIOp>(loc, lhsAff, addi.getRhs());
              }
              // Try RHS
              SmallVector<Value, 8> dimsR, symsR;
              if (AffineExpr er =
                      aeb.buildFromIndexValue(addi.getRhs(), dimsR, symsR)) {
                AffineMap mapR = AffineMap::get(dimsR.size(), symsR.size(), er);
                SmallVector<Value, 16> opsR;
                opsR.append(dimsR.begin(), dimsR.end());
                opsR.append(symsR.begin(), symsR.end());
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointAfter(def);
                Value rhsAff = b.create<affine::AffineApplyOp>(loc, mapR, opsR);
                return b.create<arith::AddIOp>(loc, addi.getLhs(), rhsAff);
              }
            } else if (auto subi = dyn_cast<arith::SubIOp>(def)) {
              if (subi.getLhs().getType().isIndex() &&
                  subi.getRhs().getType().isIndex()) {
                SmallVector<Value, 2> ds = {subi.getLhs(), subi.getRhs()};
                AffineMap m = AffineMap::get(/*dims=*/2, /*symbols=*/0,
                                             getAffineDimExpr(0, ctx) -
                                                 getAffineDimExpr(1, ctx));
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointAfter(def);
                return b.create<affine::AffineApplyOp>(loc, m, ds);
              }
              SmallVector<Value, 8> dimsL, symsL;
              if (AffineExpr el =
                      aeb.buildFromIndexValue(subi.getLhs(), dimsL, symsL)) {
                AffineMap mapL = AffineMap::get(dimsL.size(), symsL.size(), el);
                SmallVector<Value, 16> opsL;
                opsL.append(dimsL.begin(), dimsL.end());
                opsL.append(symsL.begin(), symsL.end());
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointAfter(def);
                Value lhsAff = b.create<affine::AffineApplyOp>(loc, mapL, opsL);
                return b.create<arith::SubIOp>(loc, lhsAff, subi.getRhs());
              }
              SmallVector<Value, 8> dimsR, symsR;
              if (AffineExpr er =
                      aeb.buildFromIndexValue(subi.getRhs(), dimsR, symsR)) {
                AffineMap mapR = AffineMap::get(dimsR.size(), symsR.size(), er);
                SmallVector<Value, 16> opsR;
                opsR.append(dimsR.begin(), dimsR.end());
                opsR.append(symsR.begin(), symsR.end());
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointAfter(def);
                Value rhsAff = b.create<affine::AffineApplyOp>(loc, mapR, opsR);
                return b.create<arith::SubIOp>(loc, subi.getLhs(), rhsAff);
              }
            }
          }
          return idx;
        }
        AffineMap map = AffineMap::get(dims.size(), symbols.size(), expr);
        SmallVector<Value, 16> operands;
        operands.append(dims.begin(), dims.end());
        operands.append(symbols.begin(), symbols.end());
        if (Operation *def = idx.getDefiningOp())
          b.setInsertionPointAfter(def);
        else
          b.setInsertionPointToStart(&func.front());
        Value applied = b.create<affine::AffineApplyOp>(loc, map, operands);
        return applied;
      };

      // Use exactly the last three function arguments as dims (%arg12-%arg14).
      SmallVector<Value, 3> spatialDims;
      if (func.getNumArguments() >= 3) {
        unsigned n = func.getNumArguments();
        for (unsigned i = n - 3; i < n; ++i)
          spatialDims.push_back(func.getArgument(i));
      }

      // Walk loads/stores and attempt conversion to affine ops.
      func.walk([&](Operation *op) {
        // Ensure scf.for loop bounds/step types match the IV type to avoid
        // verifier failures after we changed operand types above.
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
          Type ivTy = loop.getInductionVar().getType();
          auto castTo = [&](Value v) -> Value {
            if (v.getType() == ivTy)
              return v;
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPoint(loop);
            if (ivTy.isIndex()) {
              // Cast to index.
              if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>())
                return b.create<arith::ConstantIndexOp>(cInt.getLoc(),
                                                        cInt.value());
              if (mlir::isa<IntegerType>(v.getType()))
                return b.create<arith::IndexCastOp>(v.getLoc(), ivTy, v);
              if (auto cIdx = v.getDefiningOp<arith::ConstantIndexOp>())
                return cIdx.getResult();
              return v;
            }
            // Cast to integer type (e.g. i32) matching the IV.
            if (mlir::isa<IndexType>(v.getType()))
              return b.create<arith::IndexCastOp>(v.getLoc(), ivTy, v);
            if (auto cIdx = v.getDefiningOp<arith::ConstantIndexOp>())
              return b.create<arith::ConstantIntOp>(cIdx.getLoc(), ivTy,
                                                    cIdx.value());
            if (auto intSrc = mlir::dyn_cast<IntegerType>(v.getType())) {
              unsigned srcW = intSrc.getWidth();
              unsigned dstW = mlir::cast<IntegerType>(ivTy).getWidth();
              if (srcW == dstW)
                return v;
              if (srcW < dstW)
                return b.create<arith::ExtSIOp>(v.getLoc(), ivTy, v);
              return b.create<arith::TruncIOp>(v.getLoc(), ivTy, v);
            }
            return v;
          };
          Value lb = castTo(loop.getLowerBound());
          Value ub = castTo(loop.getUpperBound());
          Value st = castTo(loop.getStep());
          if (lb != loop.getLowerBound())
            loop.getOperation()->setOperand(0, lb);
          if (ub != loop.getUpperBound())
            loop.getOperation()->setOperand(1, ub);
          if (st != loop.getStep())
            loop.getOperation()->setOperand(2, st);
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
          LinearFormBuilder lfb(ctx, dimsAll);
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
              SmallVector<Value, 8> orderedSyms;
              AffineExpr remapped =
                  remapSymbolsSequential(e, orderedSyms, lfb.getSymbols());
              SmallVector<Value, 16> operands;
              for (Value d : lfb.getDims())
                operands.push_back(castToIndexAt(d));
              for (Value s : orderedSyms)
                operands.push_back(castToIndexAt(s));
              AffineMap m = AffineMap::get(lfb.getDims().size(),
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
            Value aff = ensureAffine(idx, load.getLoc());
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
            Value aff = ensureAffine(idx, store.getLoc());
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
          LinearFormBuilder lfb(ctx, dimsAll);
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
            // First try a pure affine.apply expression.
            if (AffineExpr e = lfb.build(val)) {
              SmallVector<Value, 8> orderedSyms;
              AffineExpr remapped =
                  remapSymbolsSequential(e, orderedSyms, lfb.getSymbols());
              SmallVector<Value, 16> operands;
              for (Value d : lfb.getDims())
                operands.push_back(castToIndexAt(d));
              for (Value s : orderedSyms)
                operands.push_back(castToIndexAt(s));
              AffineMap map = AffineMap::get(lfb.getDims().size(),
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
          LinearFormBuilder lfb(ctx, dimsAll);
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
              SmallVector<Value, 8> orderedSyms;
              AffineExpr remapped =
                  remapSymbolsSequential(e, orderedSyms, lfb.getSymbols());
              SmallVector<Value, 16> operands;
              for (Value d : lfb.getDims())
                operands.push_back(castToIndexAt(d));
              for (Value s : orderedSyms)
                operands.push_back(castToIndexAt(s));
              AffineMap map = AffineMap::get(lfb.getDims().size(),
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
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createTritonSharedAffinizePass() {
  return std::make_unique<TritonSharedAffinizePass>();
}

void registerTritonSharedAffinizePass() {
  PassRegistration<TritonSharedAffinizePass>();
}

} // namespace passes
} // namespace tmd
