/**
 * @file triton_shared_grid_to_parallel.h
 * @brief Replace grid index ABI with a 3-D `affine.parallel` loop nest.
 * @details
 * Converts the Triton-shared grid calling convention
 * `(sizeX, sizeY, sizeZ, idxX, idxY, idxZ)` into a single 3-D
 * `affine.parallel` whose IVs replace the index arguments, and erases the
 * index arguments from the function signature.
 *
 * Usage
 * - Register: `loom::passes::registerTritonSharedGridToParallelPass()`
 * - CLI: `--loom-triton-shared-grid-to-parallel`
 * - Run after `--loom-triton-shared-affinize`.
 *
 * Constraints
 * - Function must have at least 6 arguments; the last six are interpreted as
 *   `(sizeX, sizeY, sizeZ, idxX, idxY, idxZ)`.
 * - Sizes are used as dynamic upper bounds; indices are fully replaced.
 *
 * Notes
 * - The resulting parallel region becomes the anchor for spatial mapping and
 *   may be further tiled/annotated by subsequent passes.
 */

#pragma once

#include "mlir/Pass/Pass.h"

namespace loom {
namespace passes {

/**
 * \brief Create a pass that wraps function bodies with a 3-D affine.parallel
 *        and removes grid index arguments.
 *
 * Expected calling convention (from Triton-shared after affinization):
 * - Last 6 function arguments are: sizeX, sizeY, sizeZ, idxX, idxY, idxZ.
 * - The pass inserts a single three-dimensional `affine.parallel` with upper
 *   bounds (sizeX, sizeY, sizeZ) and step (1, 1, 1).
 * - All uses of (idxX, idxY, idxZ) inside the function body are replaced by the
 *   induction variables of the new parallel loop.
 * - The last three arguments (idxX, idxY, idxZ) are erased from the function
 *   signature.
 */
std::unique_ptr<mlir::Pass> createTritonSharedGridToParallelPass();

/**
 * \brief Register the grid-to-parallel pass for textual pipelines.
 */
void registerTritonSharedGridToParallelPass();

} // namespace passes
} // namespace loom
