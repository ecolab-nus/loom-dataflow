/**
 * @file enumerate_copy_broadcast.h
 * @brief Enumerate interconnect broadcast choices for loom.copy operations.
 * @details
 * This pass analyzes loom.copy operations and checks their source operations
 * for spatial reuse information from loom.reinterpret_cast operations.
 * The spatial_reuse attribute from loom.reinterpret_cast indicates whether
 * the data can be reused across spatial dimensions. It enumerates all possible
 * interconnect mapping choices (DRAM, horizontal, vertical, all-directions
 * broadcast) and generates function clones for each combination.
 */

#pragma once

#include "mlir/Pass/Pass.h"

namespace loom {
namespace passes {

/**
 * \brief Create a pass that enumerates copy interconnect broadcast choices.
 *
 * The pass walks all loom.copy operations in the module and checks if their
 * source operations (loom.reinterpret_cast) have spatial_reuse enabled.
 * It then enumerates all possible interconnect broadcast choices and generates
 * function clones for each combination.
 *
 * @param analysisOnly Currently unused, reserved for future use.
 */
std::unique_ptr<mlir::Pass>
createEnumerateCopyBroadcastPass(bool analysisOnly = false);

/** Register the pass with MLIR. */
void registerEnumerateCopyBroadcastPass();

} // namespace passes
} // namespace loom
