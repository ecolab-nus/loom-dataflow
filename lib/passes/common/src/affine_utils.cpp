#include "affine_utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace loom_affine {

// --- Tiling ---

LogicalResult tileAffineParallel(affine::AffineParallelOp original_parop,
                                 int64_t tilingFactor, unsigned tileDimIndex,
                                 TiledParallels &result) {
  MLIRContext *ctx = original_parop.getContext();
  if (tilingFactor <= 0)
    return original_parop.emitError("tiling-factor must be positive"),
           failure();

  OpBuilder builder(ctx);
  Location loc = original_parop.getLoc();
  unsigned numDims = original_parop.getNumDims();
  if (numDims == 0)
    return success();
  if (tileDimIndex >= numDims)
    return original_parop.emitError("tileDimIndex out of range"), failure();

  SmallVector<int64_t> steps = llvm::to_vector(original_parop.getSteps());
  if (steps.empty() || steps[tileDimIndex] != 1)
    return original_parop.emitError(
               "only step 1 supported on the chosen dimension"),
           failure();

  AffineMap lbChosen = original_parop.getLowerBoundMap(tileDimIndex);
  if (lbChosen.getNumResults() != 1 || !lbChosen.isSingleConstant())
    return original_parop.emitError(
               "expected constant-zero lower bound on chosen dimension"),
           failure();
  if (lbChosen.getSingleConstantResult() != 0)
    return original_parop.emitError(
               "expected lower bound 0 on chosen dimension"),
           failure();

  if (auto maybeRanges = original_parop.getConstantRanges()) {
    auto ranges = *maybeRanges;
    if (!ranges.empty()) {
      if (tileDimIndex >= ranges.size())
        return original_parop.emitError(
                   "range metadata missing for chosen dimension"),
               failure();
      int64_t extentK = ranges[tileDimIndex];
      if (extentK % tilingFactor != 0)
        return original_parop.emitError(
                   "chosen-dimension bound not divisible by tiling-factor"),
               failure();
    }
  }

  builder.setInsertionPoint(original_parop);
  IRMapping ivMap;

  SmallVector<AffineMap> tiled_new_lb_maps = {builder.getConstantAffineMap(0)};
  SmallVector<AffineMap> tiled_new_ub_maps = {
      builder.getConstantAffineMap(static_cast<int64_t>(tilingFactor))};
  SmallVector<int64_t> tiled_new_steps = {1};
  auto tiled_new = affine::AffineParallelOp::create(
      builder, loc, TypeRange{}, ArrayRef<arith::AtomicRMWKind>{},
      tiled_new_lb_maps, ValueRange{}, tiled_new_ub_maps, ValueRange{},
      tiled_new_steps);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(tiled_new.getBody());
  SmallVector<AffineMap> tiled_org_lb_maps;
  SmallVector<AffineMap> tiled_org_ub_maps;
  tiled_org_lb_maps.reserve(numDims);
  tiled_org_ub_maps.reserve(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    tiled_org_lb_maps.push_back(original_parop.getLowerBoundMap(i));
    AffineMap ubI = original_parop.getUpperBoundMap(i);
    if (i == tileDimIndex) {
      if (ubI.getNumResults() != 1)
        return original_parop.emitError("expected single-result UB map"),
               failure();
      AffineExpr e = ubI.getResult(0);
      AffineExpr transformed = e.ceilDiv(tilingFactor);
      tiled_org_ub_maps.push_back(
          AffineMap::get(ubI.getNumDims(), ubI.getNumSymbols(), transformed));
    } else {
      tiled_org_ub_maps.push_back(ubI);
    }
  }

  ValueRange org_lb_args = original_parop.getLowerBoundsOperands();
  ValueRange org_ub_args = original_parop.getUpperBoundsOperands();
  SmallVector<int64_t> org_steps = steps;

  auto tiled_org = affine::AffineParallelOp::create(
      builder, loc, TypeRange{}, ArrayRef<arith::AtomicRMWKind>{},
      tiled_org_lb_maps, org_lb_args, tiled_org_ub_maps, org_ub_args,
      org_steps);

  Block *org_body = original_parop.getBody();
  Block *tiled_org_body = tiled_org.getBody();

  for (unsigned i = 0; i < numDims; ++i) {
    org_body->getArgument(i).replaceAllUsesWith(tiled_org_body->getArgument(i));
  }

  tiled_org_body->getOperations().clear();
  tiled_org_body->getOperations().splice(tiled_org_body->begin(),
                                         org_body->getOperations());

  TilingMetadata metadata;
  metadata.originalDimIdx = tileDimIndex;
  metadata.tilingFactor = tilingFactor;
  metadata.outerIV = tiled_new.getBody()->getArgument(0);
  metadata.innerIV = tiled_org.getBody()->getArgument(tileDimIndex);

  result.tiled_org_ = tiled_org;
  result.tiled_new_ = tiled_new;
  result.tilingMetadata.push_back(metadata);

  original_parop.erase();
  return success();
}

// --- Parallel to Nested For ---

/// Recursively materialize an AffineExpr as index-typed arith ops.
/// Unlike `mlir::affine::expandAffineMap`, this emits `arith.ceildivui` /
/// `arith.divui` / `arith.remui` directly instead of the sign-safe lowering
/// (cmpi + select chains).  Trip counts and tile sizes are always
/// non-negative, so the unsigned variants are sound, and the resulting
/// def-use chain is shallow enough for `traceIndexValueToExpr` to recover
/// a symbolic `Expr`.
static Value materializeAffineExprAsIndex(OpBuilder &builder, Location loc,
                                          AffineExpr expr,
                                          ValueRange dimOperands,
                                          ValueRange symOperands) {
  if (auto c = llvm::dyn_cast<AffineConstantExpr>(expr)) {
    return arith::ConstantIndexOp::create(builder, loc, c.getValue());
  }
  if (auto d = llvm::dyn_cast<AffineDimExpr>(expr)) {
    assert(d.getPosition() < dimOperands.size() && "dim operand OOB");
    return dimOperands[d.getPosition()];
  }
  if (auto s = llvm::dyn_cast<AffineSymbolExpr>(expr)) {
    assert(s.getPosition() < symOperands.size() && "sym operand OOB");
    return symOperands[s.getPosition()];
  }
  auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
  Value lhs = materializeAffineExprAsIndex(builder, loc, bin.getLHS(),
                                           dimOperands, symOperands);
  Value rhs = materializeAffineExprAsIndex(builder, loc, bin.getRHS(),
                                           dimOperands, symOperands);
  switch (expr.getKind()) {
  case AffineExprKind::Add:
    return arith::AddIOp::create(builder, loc, lhs, rhs);
  case AffineExprKind::Mul:
    return arith::MulIOp::create(builder, loc, lhs, rhs);
  case AffineExprKind::CeilDiv:
    return arith::CeilDivUIOp::create(builder, loc, lhs, rhs);
  case AffineExprKind::FloorDiv:
    return arith::DivUIOp::create(builder, loc, lhs, rhs);
  case AffineExprKind::Mod:
    return arith::RemUIOp::create(builder, loc, lhs, rhs);
  default:
    llvm_unreachable("unexpected AffineExpr kind in trip-count materialization");
  }
}

static LogicalResult getIteratorBoundsAndStep(affine::AffineParallelOp par,
                                              unsigned dim, AffineMap &lbMap,
                                              SmallVector<Value> &lbOperands,
                                              AffineMap &ubMap,
                                              SmallVector<Value> &ubOperands,
                                              int64_t &step) {
  if (dim >= par.getNumDims())
    return failure();
  lbMap = par.getLowerBoundMap(dim);
  ubMap = par.getUpperBoundMap(dim);
  auto allLbOperands = par.getLowerBoundsOperands();
  auto allUbOperands = par.getUpperBoundsOperands();
  lbOperands.assign(allLbOperands.begin(), allLbOperands.end());
  ubOperands.assign(allUbOperands.begin(), allUbOperands.end());
  auto steps = par.getSteps();
  if (dim >= steps.size())
    return failure();
  step = steps[dim];
  return success();
}

LogicalResult ConvertParallelToNested(affine::AffineParallelOp par,
                                      ArrayRef<unsigned> order) {
  if (par.getNumResults() != 0)
    return failure();
  const unsigned P = par.getNumDims();
  if (order.size() != P)
    return failure();
  SmallVector<unsigned> seen(P, 0);
  for (unsigned v : order) {
    if (v >= P || seen[v])
      return failure();
    seen[v] = 1;
  }

  OpBuilder builder(par);
  Location loc = par.getLoc();
  MLIRContext *ctx = par.getContext();

  SmallVector<scf::ForOp> newFors;
  newFors.reserve(P);

  for (unsigned i = 0; i < P; ++i) {
    unsigned iterIdx = order[i];

    AffineMap lbMap, ubMap;
    SmallVector<Value> lbOperands, ubOperands;
    int64_t stepVal = 1;
    if (failed(getIteratorBoundsAndStep(par, iterIdx, lbMap, lbOperands, ubMap,
                                        ubOperands, stepVal)))
      return failure();

    // Only single-result lower/upper bound maps are supported by scf.for.
    // affine.parallel bound maps can be multi-result (for min/max semantics);
    // our tiling never produces such maps, but fail loudly if one appears.
    if (lbMap.getNumResults() != 1 || ubMap.getNumResults() != 1)
      return par.emitError(
          "ConvertParallelToNested: multi-result affine.parallel bound "
          "maps are not supported when lowering to scf.for");

    // Flatten nested ceildiv in the UB so the expanded arith chain stays
    // shallow (N ceildiv (A*B) instead of (N ceildiv A) ceildiv B).
    AffineExpr flatUbExpr = flattenNestedCeilDiv(ubMap.getResult(0));
    if (flatUbExpr != ubMap.getResult(0))
      ubMap = AffineMap::get(ubMap.getNumDims(), ubMap.getNumSymbols(),
                             flatUbExpr, ctx);

    if (i == 0)
      builder.setInsertionPoint(par);
    else
      builder.setInsertionPointToStart(newFors.back().getBody());

    // Materialize bound affine maps as plain index-typed arith ops.
    // affine.parallel bound operands are laid out as [dims..., syms...],
    // where the leading `numDims` operands map to AffineDimExprs and the
    // remainder map to AffineSymbolExprs.
    auto splitOperands = [](AffineMap m, ArrayRef<Value> all,
                            SmallVector<Value> &dims,
                            SmallVector<Value> &syms) {
      unsigned nd = m.getNumDims();
      dims.assign(all.begin(), all.begin() + nd);
      syms.assign(all.begin() + nd, all.end());
    };
    SmallVector<Value> lbDimOps, lbSymOps, ubDimOps, ubSymOps;
    splitOperands(lbMap, lbOperands, lbDimOps, lbSymOps);
    splitOperands(ubMap, ubOperands, ubDimOps, ubSymOps);

    Value lb = materializeAffineExprAsIndex(builder, loc, lbMap.getResult(0),
                                            lbDimOps, lbSymOps);
    Value ub = materializeAffineExprAsIndex(builder, loc, ubMap.getResult(0),
                                            ubDimOps, ubSymOps);
    Value step = arith::ConstantIndexOp::create(builder, loc, stepVal);

    scf::ForOp forOp = scf::ForOp::create(builder, loc, lb, ub, step);
    // Annotate with the loom.sym block symbol found in this dimension's UB.
    // The materialized `ub` Value traces back through arith ceildiv chains to
    // a loom.sym op whose symbol_ref names the logical tile dimension.
    if (auto blockSym = traceToLoomSymRef(ub))
      forOp->setAttr("loom.block_sym", *blockSym);
    newFors.push_back(forOp);
  }

  // Map each original parallel IV to the corresponding new scf.for IV.
  IRMapping mapping;
  SmallVector<Value> forIVForIter(P);
  for (unsigned depth = 0; depth < P; ++depth)
    forIVForIter[order[depth]] = newFors[depth].getInductionVar();
  for (unsigned iterIdx = 0; iterIdx < P; ++iterIdx)
    mapping.map(par.getIVs()[iterIdx], forIVForIter[iterIdx]);

  // Clone the parallel body into the innermost scf.for body, skipping the
  // affine.yield terminator.  scf::ForOp::create already installs an
  // scf.yield (no iter_args → no operands), which we keep as the terminator.
  scf::ForOp innermost = newFors.back();
  Block *innerBody = innermost.getBody();
  builder.setInsertionPoint(innerBody->getTerminator());

  Block &srcBody = par.getRegion().front();
  for (Operation &op : srcBody) {
    if (isa<affine::AffineYieldOp>(op))
      continue;
    builder.clone(op, mapping);
  }

  par.erase();
  return success();
}

// --- Flatten nested ceildiv ---

AffineExpr flattenNestedCeilDiv(AffineExpr expr) {
  if (!expr)
    return expr;
  if (expr.getKind() != AffineExprKind::CeilDiv)
    return expr;

  auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
  // Recursively flatten the numerator side first.
  AffineExpr lhs = flattenNestedCeilDiv(bin.getLHS());
  AffineExpr rhs = bin.getRHS();

  // If the (possibly-flattened) numerator is itself a CeilDiv, merge:
  //   (N ceildiv A) ceildiv B  -->  N ceildiv (A * B)
  // MLIR's AffineExpr multiplication folds constants automatically,
  // so  (s0 * 8) * 8  becomes  s0 * 64.
  if (lhs.getKind() == AffineExprKind::CeilDiv) {
    auto innerBin = llvm::cast<AffineBinaryOpExpr>(lhs);
    AffineExpr combinedDenom = innerBin.getRHS() * rhs;
    return innerBin.getLHS().ceilDiv(combinedDenom);
  }

  return lhs.ceilDiv(rhs);
}

// --- loom.sym tracing ---

std::optional<SymbolRefAttr> traceToLoomSymRef(Value v) {
  if (!v)
    return std::nullopt;
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  // Match loom.sym by op name to avoid pulling in loom dialect headers.
  if (defOp->getName().getStringRef() == "loom.sym") {
    if (auto attr = defOp->getAttrOfType<SymbolRefAttr>("symbol_ref"))
      return attr;
  }
  // Recurse through all operands. The caller guarantees exactly one loom.sym
  // is reachable from any given UB value, so the first hit is the right one.
  for (Value operand : defOp->getOperands()) {
    if (auto found = traceToLoomSymRef(operand))
      return found;
  }
  return std::nullopt;
}

} // namespace loom_affine
