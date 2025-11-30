/**
 * @file triton_shared_spatial_mapping_pass.h
 * @brief MLIR pass to enumerate spatial mappings for Triton-shared kernels.
 * @details
 * This pass discovers hardware spatial dimensions from DF ops in the same
 * module and enumerates all unique mappings of those dimensions onto the
 * outermost `affine.parallel` in each function. It then replaces the original
 * functions with clones for each mapping. When enabled, it also converts the
 * surviving outer `affine.parallel` into nested `affine.for` loops for all
 * iterator orders.
 */

#pragma once

#include "mlir/Pass/Pass.h"

namespace tmd {
namespace passes {

/**
 * \brief Create the Triton-shared spatial mapping exploration pass.
 *
 * The pass expects a module that contains DF declarations (e.g.,
 * `df.spatial_dim`) and one or more `func.func` kernels that have already been
 * transformed by the Triton-shared pipeline (affinize + grid_to_parallel).
 *
 * Behavior:
 * - Collect spatial dims from DF ops within the same module.
 * - Enumerate spatial mappings across all functions; when \p withOuterFors is
 *   true, also enumerate all permutations of converting the remaining outer
 *   `affine.parallel` to nested `affine.for` loops.
 * - Replace original (non-DF) top-level operations with the generated clones,
 *   inserting clones after the DF declarations so DF ops remain at the top.
 */
std::unique_ptr<mlir::Pass>
createTritonSharedExploreSpatialMappingsPass(bool withOuterFors = true);

/** Register the pass with MLIR. */
void registerTritonSharedExploreSpatialMappingsPass();

} // namespace passes
} // namespace tmd
