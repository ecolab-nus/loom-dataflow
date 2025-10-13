/**
 * @file triton_shared_affinize.h
 * @brief Affinization pass for Triton-shared lowered kernels.
 * @details
 * Converts arithmetic index expressions (arith dialect) into `affine.apply`
 * and replaces eligible `memref.load/store` with `affine.load/store` to expose
 * affine relationships. This is the entry point of the Triton → dataflow
 * pipeline and is a prerequisite for subsequent grid-to-parallel and spatial
 * mapping passes.
 *
 * Usage
 * - Register: `tmd::passes::registerTritonSharedAffinizePass()`
 * - CLI pipeline: `--tmd-triton-shared-affinize`
 * - Intended to run before `tmd-triton-shared-grid-to-parallel`.
 *
 * Preconditions
 * - Functions follow the Triton-shared ABI where the last 6 block arguments
 *   carry grid sizes and indices: `(sizeX, sizeY, sizeZ, idxX, idxY, idxZ)`.
 * - Index math is expressed in the `arith` dialect or via loop IVs.
 *
 * Semantics and Scope
 * - Treats the last six function arguments (grid sizes and indices) and loop
 *   IVs as affine dims, all other function arguments as affine symbols.
 * - Rewrites index-typed values to `affine.apply` where provably affine; keeps
 *   original values when non-affine.
 * - Converts loads/stores to affine.memrefs when all indices are affine.
 * - Rebuilds selected `memref.reinterpret_cast`, `tensor.extract_slice` and
 *   `memref.subview` operands to use affine ops when possible.
 *
 * Limitations
 * - Only linear/affine-preserving operations are recognized (add/sub, mul/div
 *   by constants, min/max with affine arguments). Non-affine subgraphs are
 *   promoted to symbols or left unchanged.
 * - Affinization of sizes/strides for `reinterpret_cast` is conservative
 *   (focuses primarily on offsets).
 */

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
