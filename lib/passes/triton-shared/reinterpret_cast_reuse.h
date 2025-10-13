#pragma once

#include "mlir/Pass/Pass.h"

namespace tmd {
namespace passes {

// Create a pass that annotates `memref.reinterpret_cast` ops with reuse
// information relative to surrounding affine/scf iterators.
std::unique_ptr<mlir::Pass> createAnnotateReinterpretCastReusePass();

// Register the reinterpret-cast reuse annotation pass with MLIR.
void registerAnnotateReinterpretCastReusePass();

} // namespace passes
} // namespace tmd
