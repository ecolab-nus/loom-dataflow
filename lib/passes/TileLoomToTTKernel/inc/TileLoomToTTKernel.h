/**
 * @file TileLoomToTTKernel.h
 * @brief Header for TileLoom to TTKernel conversion pass.
 */

#ifndef LOOM_PASSES_TILELOOMTOTTKERNEL_TILELOOMTOTTKERNEL_H
#define LOOM_PASSES_TILELOOMTOTTKERNEL_TILELOOMTOTTKERNEL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace loom {

/**
 * @brief Create the TileLoom to TTKernel conversion pass.
 * @return A unique pointer to the pass.
 */
std::unique_ptr<Pass> createTileLoomToTTKernelPass();

/**
 * @brief Register the TileLoom to TTKernel conversion pass.
 * 
 * @details This function registers the pass so it can be used with mlir-opt
 *          or other MLIR tools that support pass registration.
 */
void registerTileLoomToTTKernelPass();

} // namespace loom
} // namespace mlir

#endif // LOOM_PASSES_TILELOOMTOTTKERNEL_TILELOOMTOTTKERNEL_H

