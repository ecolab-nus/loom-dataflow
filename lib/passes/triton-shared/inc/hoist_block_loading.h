#pragma once

#include <mlir/Pass/Pass.h>

namespace loom {
namespace passes {

/** Create the reuse-annotation pass. */
std::unique_ptr<mlir::Pass> createHoistBlockLoadingPass();

/** Register the reuse-annotation pass for textual pipelines. */
void registerHoistBlockLoadingPass();

} // namespace passes
} // namespace loom