// Transform Triton-shared kernels by replacing grid index arguments with
// top-level affine.parallel loops.
//
// This pass assumes the last six arguments of each `func.func` represent the
// grid: three sizes followed by three indices. It creates a 3-D
// `affine.parallel` whose upper bounds are the three size arguments and whose
// induction variables replace the three index arguments within the function
// body. Finally, it erases the three index arguments from the function
// signature.
//
// Note: The three size arguments are preserved as dynamic upper bounds for the
// loops. If they are not of index type, they are cast to index as needed. The
// resulting `affine.parallel` serves as the anchor point for spatial mappings:
// later exploration passes match its induction variables with hardware
// dimensions declared in the `df` dialect and may wrap the parallel loop in
// additional `affine.for` nests to model sequential waves across the mesh.

#pragma once

#include "mlir/Pass/Pass.h"

namespace tmd {
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
} // namespace tmd
