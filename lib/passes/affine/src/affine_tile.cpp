/**
 * @file
 * @brief Tiling utilities for `mlir::affine::AffineParallelOp`.
 *
 * @details Implements tiling of an `affine.parallel` by a constant factor on a
 * chosen dimension. The transformation replaces the original loop with:
 * - an outer `affine.parallel` that iterates `[0, tilingFactor)` with unit
 *   step, and
 * - an inner `affine.parallel` whose upper bound on the tiled dimension is
 *   `ceilDiv(originalUB, tilingFactor)`.
 *
 * The original induction variable on the tiled dimension is remapped to
 * `outerIv * tilingFactor + innerIv`. Non-tiled dimensions reuse the inner
 * loop's induction variables. The original body is cloned into the inner loop, and the
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

namespace loom_affine {
/// Affine utilities and transformations specific to LOOM.

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
 *   - Create an outer `affine.parallel` with range `[0, tilingFactor)`.
 *   - Create an inner `affine.parallel` reusing the original bounds on all
 *     dimensions except the chosen one, for which the upper bound becomes
 *     `ceilDiv(originalUB, tilingFactor)`.
 *   - Remap induction variables: on the chosen dimension,
 *     `oldIv -> outerIv * tilingFactor + innerIv`; other dimensions map to the
 *     corresponding inner IVs.
 *   - Clone the original body into the inner loop and erase the original op.
 */
LogicalResult tileAffineParallel(affine::AffineParallelOp original_parop,
                                 int64_t tilingFactor, unsigned tileDimIndex,
                                 TiledParallels &result) {
  MLIRContext *ctx = original_parop.getContext();
  if (tilingFactor <= 0)
    return original_parop.emitError("tiling-factor must be positive"), failure();

  OpBuilder builder(ctx);
  Location loc = original_parop.getLoc();
  unsigned numDims = original_parop.getNumDims();
  if (numDims == 0)
    return success();
  if (tileDimIndex >= numDims)
    return original_parop.emitError("tileDimIndex out of range"), failure();

  // The tiled dimension must have unit step for this implementation.
  SmallVector<int64_t> steps = llvm::to_vector(original_parop.getSteps());
  if (steps.empty() || steps[tileDimIndex] != 1)
    return original_parop.emitError("only step 1 supported on the chosen dimension"),
           failure();

  // The tiled dimension must start from a constant-zero lower bound.
  AffineMap lbChosen = original_parop.getLowerBoundMap(tileDimIndex);
  if (lbChosen.getNumResults() != 1 || !lbChosen.isSingleConstant())
    return original_parop.emitError(
               "expected constant-zero lower bound on chosen dimension"),
           failure();
  if (lbChosen.getSingleConstantResult() != 0)
    return original_parop.emitError("expected lower bound 0 on chosen dimension"),
           failure();

  // If a static range is known, verify divisibility by the tiling factor.
  if (auto maybeRanges = original_parop.getConstantRanges()) {
    auto ranges = *maybeRanges;
    if (!ranges.empty()) {
      if (tileDimIndex >= ranges.size())
        return original_parop.emitError("range metadata missing for chosen dimension"),
               failure();
      int64_t extentK = ranges[tileDimIndex];
      if (extentK % tilingFactor != 0)
        return original_parop.emitError(
                   "chosen-dimension bound not divisible by tiling-factor"),
               failure();
    }
  }

  builder.setInsertionPoint(original_parop);

  // Prepare IV remapping.
  IRMapping ivMap;

  // Create the outer affine.parallel with constant range [0, tilingFactor).
  SmallVector<AffineMap> tiled_new_lb_maps = {builder.getConstantAffineMap(0)};
  SmallVector<AffineMap> tiled_new_ub_maps = {
      builder.getConstantAffineMap(static_cast<int64_t>(tilingFactor))};
  SmallVector<int64_t> tiled_new_steps = {1};
  auto tiled_new = builder.create<affine::AffineParallelOp>(
      loc, TypeRange{},
      ArrayRef<arith::AtomicRMWKind>{}, 
      tiled_new_lb_maps, ValueRange{}, 
      tiled_new_ub_maps, ValueRange{}, 
      tiled_new_steps
    );

  // Reuse original LB/UB maps and operands. Replace only the chosen dimension's
  // UB with ceilDiv(originalUB, tilingFactor) to form the inner loop space.
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
      // Transform the single-result UB map to ceilDiv(expr, tilingFactor).
      if (ubI.getNumResults() != 1)
        return original_parop.emitError("expected single-result UB map"), failure();
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
  // Steps remain unchanged on all inner-loop dimensions.
  SmallVector<int64_t> org_steps = steps;

  // Create the inner affine.parallel.
  auto tiled_org = builder.create<affine::AffineParallelOp>(
      loc, TypeRange{},
      ArrayRef<arith::AtomicRMWKind>{}, 
      tiled_org_lb_maps, org_lb_args, 
      tiled_org_ub_maps, org_ub_args,
      org_steps
    );

  // Clone original body ops into the inner parallel, remapping IVs. Skip the
  // terminator if present.
  Block *org_body = original_parop.getBody();
  Block *tiled_new_body = tiled_new.getBody();
  Block *tiled_org_body = tiled_org.getBody();
  // Set insertion point to the inner parallel's body start to clone the body into it.
  builder.setInsertionPointToStart(tiled_org_body);
  // Build the IV mapping now that both IVs exist.
  Value tiled_new_iv = tiled_new_body->getArgument(0);
  for (unsigned i = 0; i < numDims; ++i) {
    Value org_iv = org_body->getArgument(i);
    if (i == tileDimIndex) {
      Value tiled_org_iv = tiled_org_body->getArgument(i);
      AffineExpr d0 = getAffineDimExpr(0, ctx);
      AffineExpr d1 = getAffineDimExpr(1, ctx);
      AffineExpr cT = getAffineConstantExpr(tilingFactor, ctx);
      AffineMap tileMap =
          AffineMap::get(/*dims=*/2, /*symbols=*/0, d0 * cT + d1, ctx);
      Value combined = builder.create<affine::AffineApplyOp>(
          loc, tileMap, ValueRange{tiled_new_iv, tiled_org_iv});
      ivMap.map(org_iv, combined);
    } else {
      ivMap.map(org_iv, tiled_org_body->getArgument(i));
    }
  }
  for (Operation &nested : llvm::make_early_inc_range(*org_body)) {
    if (nested.hasTrait<OpTrait::IsTerminator>())
      continue;
    (void)builder.clone(nested, ivMap);
  }

  // Erase the original op; the rewrite is complete.
  original_parop.erase();
  // Populate result handles.
  result.tiled_org_ = tiled_org;
  result.tiled_new_ = tiled_new;
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

} // namespace loom_affine
