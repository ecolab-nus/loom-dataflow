/**
 * @file explore_alloc_copy_mapping.h
 * @brief Explore mapping choices for memref.alloc and memref.copy.
 * @details
 * Annotates `memref.alloc` with local memory placement and explores how
 * `memref.copy` operations are realized on the dataflow fabric: local
 * memory copies versus broadcasts along available interconnects. Uses
 * `tmd.reuse` metadata from `memref.reinterpret_cast` to determine broadcast
 * eligibility along spatial dimensions.
 *
 * Modes
 * - Analysis-only: attach `tmd.copy.candidates` arrays in place.
 * - Enumeration: clone per cross-product of copy-site candidates, setting a
 *   single `tmd.copy.choice` per site and suffixing function names to reflect
 *   choices.
 */

#pragma once

#include "mlir/Pass/Pass.h"

namespace tmd {
namespace passes {

/**
 * \brief Create a pass that explores alloc/copy mapping choices.
 *
 * The pass assumes a DF module declaring a single local memory (df.memory)
 * and optional memory-to-memory interconnects (df.interconnects) over named
 * spatial dimensions (df.spatial_dim). It uses prior `tmd.reuse` annotations
 * on `memref.reinterpret_cast` to determine which copies are broadcastable.
 *
 * Behavior:
 * - Always annotate `memref.alloc` with `{ tmd.alloc = { local = true,
 *   memory_name = <singleMem> } }`.
 * - For each `memref.copy`, build candidates consisting of a local memory
 *   copy and, when spatial total-reuse exists, broadcast candidates for each
 *   eligible interconnect along the reused dimension.
 * - If analysisOnly is true, attach `tmd.copy.candidates` (no cloning).
 * - Otherwise, enumerate the cross product of per-copy candidates, clone the
 *   function per combination, and attach `tmd.copy.choice` on each copy.
 *
 * @param analysisOnly When true, annotate candidates only (no clones).
 */
std::unique_ptr<mlir::Pass>
createExploreAllocCopyMappingPass(bool analysisOnly = false);

/** Register the pass with MLIR. */
void registerExploreAllocCopyMappingPass();

} // namespace passes
} // namespace tmd
