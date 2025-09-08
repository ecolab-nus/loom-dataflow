// Affinization pass for Triton-shared lowered kernels.
//
// This pass rewrites arithmetic index computations into affine.apply
// operations and converts eligible memory operations to affine form, exposing
// affine relationships among indexing variables. The last six function
// arguments that encode GPU grid/thread sizes and IDs are modeled as symbols
// and dimensions within affine maps where applicable.
//
// The pass targets code patterns produced by the Triton-shared lowering, where
// index arithmetic is typically expressed in the arith dialect before
// bufferization and Linalg operations.

#pragma once

#include "mlir/Pass/Pass.h"

namespace tmd {
namespace passes {

/**
 * \brief Create a pass that rewrites Triton-shared kernels to affine form.
 *
 * The pass attempts to:
 * - Identify arithmetic expressions that compute memref offsets/indices and
 *   replace them with `affine.apply` using an affine map built from the
 *   expression graph.
 * - Replace `memref.load`/`memref.store` with `affine.load`/`affine.store`
 *   when all indices are provably affine of loop IVs, function arguments, and
 *   constants.
 * - Where applicable, express `memref.reinterpret_cast` offsets as
 *   `affine.apply` and thread them to affine.memref operations.
 * - Treat the last six function arguments that represent the GPU grid/thread
 *   sizes and IDs as symbols/dimensions for affine map construction.
 */
std::unique_ptr<mlir::Pass> createTritonSharedAffinizePass();

/**
 * \brief Register the affinization pass for textual pipelines.
 */
void registerTritonSharedAffinizePass();

} // namespace passes
} // namespace tmd
