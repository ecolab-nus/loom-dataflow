/**
 * @file analyze_reuse.h
 * @brief Annotate `memref.reinterpret_cast` with reuse metadata.
 * @details
 * The pass walks `memref.reinterpret_cast` and attaches a `loom.reuse`
 * dictionary that summarizes how the cast's dynamic offsets depend on
 * surrounding iterators:
 * - spatial: entries for IVs of innermost-to-outermost `affine.parallel`
 * - temporal: entries for `affine.for` IVs (sequential waves)
 * - sequential: entries for `scf.for` IVs (per-core tiles)
 * Each entry contains: `iterator` (SSA name), `depth` (outermost=0),
 * `reuse_type` (`total_reuse` or `no_reuse`), `volume` (bytes, -1 unknown),
 * and optional `mapped_to` (spatial dim name from `loom.mapped_to`).
 *
 * Usage
 * - Register: `loom::passes::registerAnnotateReinterpretCastReusePass()`
 * - CLI: `--loom-annotate-reinterpret-cast-reuse`
 */

#pragma once

#include "mlir/Pass/Pass.h"

namespace loom {
namespace passes {

/** Create the reuse-annotation pass. */
std::unique_ptr<mlir::Pass> createAnnotateReinterpretCastReusePass();

/** Register the reuse-annotation pass for textual pipelines. */
void registerAnnotateReinterpretCastReusePass();

} // namespace passes
} // namespace loom
