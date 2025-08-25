#include "affine_tile.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h" // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
// IWYU: keep builtin includes for context setup in test drivers
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

namespace tmd_affine {

LogicalResult tileAffineParallel(affine::AffineParallelOp op,
                                 int64_t tilingFactor, unsigned tileDimIndex) {
  MLIRContext *ctx = op.getContext();
  if (tilingFactor <= 0)
    return op.emitError("tiling-factor must be positive"), failure();

  OpBuilder builder(ctx);
  Location loc = op.getLoc();
  unsigned numDims = op.getNumDims();
  if (numDims == 0)
    return success();
  if (tileDimIndex >= numDims)
    return op.emitError("tileDimIndex out of range"), failure();

  // Only support unit step on the chosen dimension for simplicity.
  SmallVector<int64_t> steps = llvm::to_vector(op.getSteps());
  if (steps.empty() || steps[tileDimIndex] != 1)
    return op.emitError("only step 1 supported on the chosen dimension"),
           failure();

  // Require lower bound 0 on the chosen dimension for simplicity.
  AffineMap lbChosen = op.getLowerBoundMap(tileDimIndex);
  if (lbChosen.getNumResults() != 1 || !lbChosen.isSingleConstant())
    return op.emitError(
               "expected constant-zero lower bound on chosen dimension"),
           failure();
  if (lbChosen.getSingleConstantResult() != 0)
    return op.emitError("expected lower bound 0 on chosen dimension"),
           failure();

  // If constant ranges known, verify divisibility; otherwise assume
  // divisible.
  if (auto maybeRanges = op.getConstantRanges()) {
    auto ranges = *maybeRanges;
    if (!ranges.empty()) {
      if (tileDimIndex >= ranges.size())
        return op.emitError("range metadata missing for chosen dimension"),
               failure();
      int64_t extentK = ranges[tileDimIndex];
      if (extentK % tilingFactor != 0)
        return op.emitError(
                   "chosen-dimension bound not divisible by tiling-factor"),
               failure();
    }
  }

  builder.setInsertionPoint(op);

  // Reuse original LB/UB maps and operands. Replace only the chosen UB map
  // with ceilDiv(originalUB, tilingFactor) for composable tiling.
  SmallVector<AffineMap> outerLbMaps;
  SmallVector<AffineMap> outerUbMaps;
  outerLbMaps.reserve(numDims);
  outerUbMaps.reserve(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    outerLbMaps.push_back(op.getLowerBoundMap(i));
    AffineMap ubI = op.getUpperBoundMap(i);
    if (i == tileDimIndex) {
      // Transform the single-result UB map: ceilDiv(expr, tilingFactor).
      if (ubI.getNumResults() != 1)
        return op.emitError("expected single-result UB map"), failure();
      AffineExpr e = ubI.getResult(0);
      AffineExpr transformed = e.ceilDiv(tilingFactor);
      outerUbMaps.push_back(
          AffineMap::get(ubI.getNumDims(), ubI.getNumSymbols(), transformed));
    } else {
      outerUbMaps.push_back(ubI);
    }
  }
  ValueRange outerLbArgs = op.getLowerBoundsOperands();
  ValueRange outerUbArgs = op.getUpperBoundsOperands();

  // Steps remain the same for outer loop.
  SmallVector<int64_t> outerSteps = steps;

  // Only support non-reduction parallel loops for now.
  if (!op->getResults().empty())
    return op.emitError("reduction results not supported by this tiling pass"),
           failure();

  // Create the outer affine.parallel.
  auto outerPar = builder.create<affine::AffineParallelOp>(
      loc, /*resultTypes=*/TypeRange{},
      /*reductions=*/ArrayRef<arith::AtomicRMWKind>{}, outerLbMaps,
      /*lbArgs=*/outerLbArgs, outerUbMaps,
      /*ubArgs=*/outerUbArgs, outerSteps);

  // Prepare remapping for IVs. For the tiled dimension k, map old iv_k to
  // affine.apply (d0, d1) -> (d0 * tile + d1), where d0 = outer iv_k and
  // d1 = inner iv_0. Other dims map to their corresponding outer IVs.
  IRMapping ivMap;

  // Create the inner affine.parallel with constant range (0, tilingFactor).
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(outerPar.getBody());
  (void)builder.create<affine::AffineParallelOp>(
      loc, /*resultTypes=*/TypeRange{},
      /*reductions=*/ArrayRef<arith::AtomicRMWKind>{},
      /*ranges=*/ArrayRef<int64_t>{tilingFactor});
  // Splice original body operations into the inner parallel, remapping IVs.
  // Skip the terminator if present.
  Block *origBody = op.getBody();
  Block *innerBody = outerPar.getBody();
  // The newly created innerPar is the first op in the body; set insertion
  // point to its body start to clone the body into it.
  auto &firstOp = innerBody->front();
  auto innerParOp = dyn_cast<affine::AffineParallelOp>(&firstOp);
  Block *innerParBody = innerParOp.getBody();
  builder.setInsertionPointToStart(innerParBody);
  // Build the mapping now that we have the inner IV.
  Value innerIv = innerParOp.getBody()->getArgument(0);
  for (unsigned i = 0; i < numDims; ++i) {
    Value oldIv = op.getBody()->getArgument(i);
    if (i == tileDimIndex) {
      Value outerIv = outerPar.getBody()->getArgument(i);
      AffineExpr d0 = getAffineDimExpr(0, ctx);
      AffineExpr d1 = getAffineDimExpr(1, ctx);
      AffineExpr cT = getAffineConstantExpr(tilingFactor, ctx);
      AffineMap tileMap =
          AffineMap::get(/*dims=*/2, /*symbols=*/0, d0 * cT + d1, ctx);
      Value combined = builder.create<affine::AffineApplyOp>(
          loc, tileMap, ValueRange{outerIv, innerIv});
      ivMap.map(oldIv, combined);
    } else {
      ivMap.map(oldIv, outerPar.getBody()->getArgument(i));
    }
  }
  for (Operation &nested : llvm::make_early_inc_range(*origBody)) {
    if (nested.hasTrait<OpTrait::IsTerminator>())
      continue;
    (void)builder.clone(nested, ivMap);
  }

  // Erase the original op.
  op.erase();
  return success();
}

} // namespace tmd_affine
