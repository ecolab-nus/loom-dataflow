#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"

namespace tmd_affine {

/// Tile a single affine.parallel operation along a chosen iterator by a given
/// tiling factor and rewrite it into a perfectly nested pair of
/// affine.parallel operations.
mlir::LogicalResult tileAffineParallel(mlir::affine::AffineParallelOp op,
                                       int64_t tilingFactor,
                                       unsigned tileDimIndex);

/// Create a pass that tiles the first affine.parallel in each function.
std::unique_ptr<mlir::Pass> createAffineTilePass(int64_t tilingFactor,
                                                 unsigned tileDimIndex);

} // namespace tmd_affine
