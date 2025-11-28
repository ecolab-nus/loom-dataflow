/**
 * @file
 * @brief Tiling utilities for `mlir::affine::AffineParallelOp`.
 *
 * @details Implements tiling of an `affine.parallel` by a constant factor on a
 * chosen dimension. The transformation replaces the original loop with:
 * - an outer `affine.parallel` whose upper bound on the tiled dimension is
 *   `ceilDiv(originalUB, tilingFactor)`, and
 * - an inner `affine.parallel` that iterates `[0, tilingFactor)` with unit
 * step.
 *
 * The original induction variable on the tiled dimension is remapped to
 * `outerIv * tilingFactor + innerIv`. Non-tiled dimensions reuse the outer
 * induction variables. The original body is cloned into the inner loop, and the
 * original operation is erased.
 *
 * Constraints enforced by the implementation:
 * - tiling factor must be positive
 * - only unit step on the tiled dimension is supported
 * - lower bound on the tiled dimension must be constant zero
 * - reductions (i.e., ops with results) are not supported
 * - if static ranges are known, the chosen dimension's extent must be divisible
 *   by the tiling factor
 */
#include "affine_tile.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h" // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
// IWYU: keep builtin includes for context setup in test drivers
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

namespace tmd_affine {
/// Affine utilities and transformations specific to TMD.

/**
 * @brief Tile an `affine.parallel` along one dimension, producing outer/inner
 * loops.
 *
 * @param op The `affine.parallel` to tile.
 * @param tilingFactor Tile size to apply on the chosen dimension. Must be > 0.
 * @param tileDimIndex Zero-based index of the dimension to tile.
 * @param[out] result If provided, receives the created outer and inner
 *                    `AffineParallelOp` handles.
 * @return `success()` on a successful rewrite; `failure()` with an emitted
 *         diagnostic if preconditions are not met or an internal invariant is
 *         violated.
 *
 * @details Semantics:
 * - Preconditions enforced at runtime:
 *   - positive `tilingFactor`
 *   - the chosen dimension has unit step and constant-zero lower bound
 *   - no reductions (op has no results)
 *   - when available, static extent of the chosen dimension is divisible by
 *     `tilingFactor`
 * - Rewrite:
 *   - Create an outer `affine.parallel` reusing the original bounds on all
 *     dimensions except the chosen one, for which the upper bound becomes
 *     `ceilDiv(originalUB, tilingFactor)`.
 *   - Create an inner `affine.parallel` with range `[0, tilingFactor)`.
 *   - Remap induction variables: on the chosen dimension,
 *     `oldIv -> outerIv * tilingFactor + innerIv`; other dimensions map to the
 *     corresponding outer IVs.
 *   - Clone the original body into the inner loop and erase the original op.
 */
LogicalResult tileAffineParallel(affine::AffineParallelOp op,
                                 int64_t tilingFactor, unsigned tileDimIndex,
                                 TiledParallels &result) {
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

  // The tiled dimension must have unit step for this implementation.
  SmallVector<int64_t> steps = llvm::to_vector(op.getSteps());
  if (steps.empty() || steps[tileDimIndex] != 1)
    return op.emitError("only step 1 supported on the chosen dimension"),
           failure();

  // The tiled dimension must start from a constant-zero lower bound.
  AffineMap lbChosen = op.getLowerBoundMap(tileDimIndex);
  if (lbChosen.getNumResults() != 1 || !lbChosen.isSingleConstant())
    return op.emitError(
               "expected constant-zero lower bound on chosen dimension"),
           failure();
  if (lbChosen.getSingleConstantResult() != 0)
    return op.emitError("expected lower bound 0 on chosen dimension"),
           failure();

  // If a static range is known, verify divisibility by the tiling factor.
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

  // Reuse original LB/UB maps and operands. Replace only the chosen dimension's
  // UB with ceilDiv(originalUB, tilingFactor) to form the outer loop space.
  SmallVector<AffineMap> outerLbMaps;
  SmallVector<AffineMap> outerUbMaps;
  outerLbMaps.reserve(numDims);
  outerUbMaps.reserve(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    outerLbMaps.push_back(op.getLowerBoundMap(i));
    AffineMap ubI = op.getUpperBoundMap(i);
    if (i == tileDimIndex) {
      // Transform the single-result UB map to ceilDiv(expr, tilingFactor).
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

  // Steps remain unchanged on all outer-loop dimensions.
  SmallVector<int64_t> outerSteps = steps;

  // Reductions (parallel ops with results) are not supported.
  if (!op->getResults().empty())
    return op.emitError("reduction results not supported by this tiling pass"),
           failure();

  // Create the outer affine.parallel.
  auto outerPar = builder.create<affine::AffineParallelOp>(
      loc, /*resultTypes=*/TypeRange{},
      /*reductions=*/ArrayRef<arith::AtomicRMWKind>{}, outerLbMaps,
      /*lbArgs=*/outerLbArgs, outerUbMaps,
      /*ubArgs=*/outerUbArgs, outerSteps);

  // Prepare IV remapping. For the tiled dimension k, map
  //   old_iv_k -> affine.apply (d0, d1) -> (d0 * tile + d1),
  // where d0 = outer iv_k and d1 = inner iv_0. Other dimensions map to their
  // corresponding outer IVs.
  IRMapping ivMap;

  // Create the inner affine.parallel with constant range [0, tilingFactor).
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(outerPar.getBody());
  // Create inner with explicit affine lb/ub maps: (0) to (tilingFactor).
  SmallVector<AffineMap> innerLbMaps = {builder.getConstantAffineMap(0)};
  SmallVector<AffineMap> innerUbMaps = {
      builder.getConstantAffineMap(static_cast<int64_t>(tilingFactor))};
  SmallVector<int64_t> innerSteps = {1};
  (void)builder.create<affine::AffineParallelOp>(
      loc, /*resultTypes=*/TypeRange{},
      /*reductions=*/ArrayRef<arith::AtomicRMWKind>{}, innerLbMaps,
      /*lbArgs=*/ValueRange{}, innerUbMaps,
      /*ubArgs=*/ValueRange{}, innerSteps);
  // Clone original body ops into the inner parallel, remapping IVs. Skip the
  // terminator if present.
  Block *origBody = op.getBody();
  Block *innerBody = outerPar.getBody();
  // The newly created innerPar is the first op in the body; set insertion
  // point to its body start to clone the body into it.
  auto &firstOp = innerBody->front();
  auto innerParOp = dyn_cast<affine::AffineParallelOp>(&firstOp);
  if (!innerParOp)
    return op.emitError("internal error: expected inner AffineParallelOp"),
           failure();
  Block *innerParBody = innerParOp.getBody();
  builder.setInsertionPointToStart(innerParBody);
  // Build the IV mapping now that the inner IV exists.
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

  // Erase the original op; the rewrite is complete.
  op.erase();
  // Populate result handles.
  result.original = outerPar;
  result.tiled = innerParOp;
  return success();
}

/**
 * @brief Convenience overload that discards the created loop handles.
 *
 * @param op The `affine.parallel` to tile.
 * @param tilingFactor Tile size to apply on the chosen dimension.
 * @param tileDimIndex Zero-based index of the dimension to tile.
 * @return `success()` on success; `failure()` with diagnostic otherwise.
 *
 * @see tileAffineParallel(affine::AffineParallelOp, int64_t, unsigned,
 *      TiledParallels &)
 */
LogicalResult tileAffineParallel(affine::AffineParallelOp op,
                                 int64_t tilingFactor, unsigned tileDimIndex) {
  TiledParallels tmp;
  return tileAffineParallel(op, tilingFactor, tileDimIndex, tmp);
}

} // namespace tmd_affine
