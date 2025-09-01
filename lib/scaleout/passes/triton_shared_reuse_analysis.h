#pragma once

#include "mlir/Pass/Pass.h"

namespace tmd {
namespace passes {

/**
 * \brief Create a pass that annotates memref arguments with invariance flags.
 *
 * The pass scans Triton-shared lowered kernels and marks function memref block
 * arguments with `tmd.invariant.x` and `tmd.invariant.y` boolean attributes to
 * indicate whether the argument's accessed region is invariant across the
 * corresponding spatial axes.
 */
std::unique_ptr<mlir::Pass> createTritonSharedSpatialReuseAnalysisPass();

/**
 * \brief Register the Triton-shared spatial reuse analysis pass by name.
 */
void registerTritonSharedSpatialReuseAnalysisPass();

} // namespace passes
} // namespace tmd
