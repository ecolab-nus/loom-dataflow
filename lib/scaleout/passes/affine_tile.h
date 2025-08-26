#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"

namespace tmd_affine {

/**
 * Tile a single `affine.parallel` operation along a chosen iterator by a
 * fixed factor and rewrite it into a perfectly nested pair of
 * `affine.parallel` operations.
 *
 * Transform overview (for the iterator at index `tileDimIndex`):
 * - Create an outer `affine.parallel` with the same number of iterators as the
 *   original. The upper bound map at the tiled dimension becomes
 *   `ceildiv(originalUB, tilingFactor)` using an `affine_map`; other bounds and
 *   steps are preserved.
 * - Create a 1-D inner `affine.parallel` with explicit constant bounds `(0,
 *   tilingFactor)` and step `1`.
 * - Remap the original IV at the tiled dimension `k` to an `affine.apply` of
 *   the form `(outer_k, inner_0) -> outer_k * tilingFactor + inner_0`. All
 *   other IVs are remapped to their corresponding outer IVs.
 * - The original body is cloned into the inner loop; the original op is erased.
 *
 * Preconditions/assumptions:
 * - Only unit step (1) is supported on the chosen dimension.
 * - The chosen dimension's lower bound must be the constant `0`.
 * - Reductions (i.e., `affine.parallel` results) are not supported.
 * - If constant range metadata is available on the op, divisibility checks are
 *   performed to validate `tilingFactor`; otherwise, the transform assumes it
 *   is valid.
 *
 * Composability:
 * - The transform is designed to be composable: repeatedly tiling the same
 *   dimension produces nested inner loops of size `tilingFactor`, and index
 *   expressions are formed via composition of `affine.apply` operations.
 */
struct TiledParallels {
  mlir::affine::AffineParallelOp outer;
  mlir::affine::AffineParallelOp inner;
};

/**
 * Tiles the given parallel loop and returns the newly created outer and inner
 * `affine.parallel` operations via `result` on success. See the documentation
 * above for transformation details and preconditions.
 */
mlir::LogicalResult tileAffineParallel(mlir::affine::AffineParallelOp op,
                                       int64_t tilingFactor,
                                       unsigned tileDimIndex,
                                       TiledParallels &result);

// Backward-compatible overload (result is ignored).
mlir::LogicalResult tileAffineParallel(mlir::affine::AffineParallelOp op,
                                       int64_t tilingFactor,
                                       unsigned tileDimIndex);

/**
 * Tuple describing one tiling choice: which iterator to tile and the factor.
 */
struct TileChoice {
  unsigned iteratorIndex;
  int64_t factor;
};

/// Create a pass that tiles the first affine.parallel in each function.
std::unique_ptr<mlir::Pass> createAffineTilePass(int64_t tilingFactor,
                                                 unsigned tileDimIndex);

} // namespace tmd_affine
