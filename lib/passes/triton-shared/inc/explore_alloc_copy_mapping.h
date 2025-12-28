/**
 * @file explore_alloc_copy_mapping.h
 * @brief Explore mapping choices for loom.copy operations.
 * @details
 * This pass analyzes loom.copy operations and checks their source operations
 * for spatial reuse information from loom.reinterpret_cast operations.
 * The spatial_reuse attribute from loom.reinterpret_cast indicates whether
 * the data can be reused across spatial dimensions.
 */

#pragma once

#include "mlir/Pass/Pass.h"

namespace loom {
namespace passes {

/**
 * \brief Create a pass that explores alloc/copy mapping choices.
 *
 * The pass walks all loom.copy operations in the module and checks if their
 * source operations (loom.reinterpret_cast) have spatial_reuse enabled.
 *
 * @param analysisOnly Currently unused, reserved for future use.
 */
std::unique_ptr<mlir::Pass>
createExploreAllocCopyMappingPass(bool analysisOnly = false);

/** Register the pass with MLIR. */
void registerExploreAllocCopyMappingPass();

} // namespace passes
} // namespace loom
