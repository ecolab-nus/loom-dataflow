/**
 * @file tile_scf_for_to_l1.h
 * @brief Tiling of scf.for loops to fit within the single df.memory (L1).
 * @details
 * The pass verifies that the module declares exactly one `df.memory` and that
 * all `memref.alloc` operations are annotated with `tmd.alloc = { local=true,
 * memory_name = <that memory's label> }`. For each `scf.for` loop, it estimates
 * the per-iteration memory footprint as the sum of byte sizes of all static
 * `memref.alloc` inside the loop body. It then picks the largest power-of-two
 * tile factor such that `factor * perIterationBytes <= L1SizeBytes` and checks
 * the loop trip count is a multiple of this factor. If any requirement cannot
 * be proven, the pass fails.
 *
 * Transformation:
 * - Rewrites each eligible `scf.for` into an outer `affine.for` over tiles and
 *   an inner `scf.for` that iterates within the tile with the original step.
 * - Keeps the inner loop as `scf.for` and materializes the outer loop as
 *   `affine.for`.
 *
 * Tiling semantics used by this pass (conceptual):
 * - Given an original loop `scf.for %i = lb to ub step s`, and a tiling factor
 *   `f`, the inner loop performs exactly `f` iterations with step `s`, i.e.
 *   it ranges over an extent of `f * s` in the induction variable space.
 * - The outer loop enumerates tiles; conceptually it runs from `0` to
 *   `ceildiv(N, f)` where `N = (ub - lb) / s` is the trip count.
 * - The original loop index at iteration (tile t, inner k) is computed as
 *   `i = lb + (t * f + k) * s`.
 * - Implementation note: this pass currently requires that `N` is provably
 *   divisible by `f` and fails otherwise; under that assumption,
 *   `ceildiv(N, f)` becomes a plain integer division.
 */

#pragma once

#include "mlir/Pass/Pass.h"

namespace tmd {
namespace passes {

/**
 * \brief Create a pass that tiles `scf.for` loops to fit the single df.memory.
 */
std::unique_ptr<mlir::Pass> createTileScfForToL1Pass();

/**
 * \brief Register the tiling pass for textual pipelines.
 */
void registerTileScfForToL1Pass();

} // namespace passes
} // namespace tmd
