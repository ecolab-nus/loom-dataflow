/**
 * @file loom_to_memref.h
 * @brief Lowering pass for converting loom dialect operations to memref dialect.
 * @details
 * This pass lowers loom.reinterpret_cast and loom.copy operations to their
 * corresponding memref dialect operations. It operates on the module level
 * and converts loom-specific operations to standard memref operations.
 *
 * Usage
 * - Register: `loom::passes::registerLoomToMemRefLoweringPass()`
 * - CLI: `--loom-to-memref`
 */

#pragma once

#include "mlir/Pass/Pass.h"

namespace loom {
namespace passes {

/**
 * \brief Create a pass that lowers loom dialect operations to memref dialect.
 *
 * This pass performs lowering transformations to convert loom.reinterpret_cast
 * and loom.copy operations to standard memref dialect operations.
 */
std::unique_ptr<mlir::Pass> createLoomToMemRefLoweringPass();

/**
 * \brief Register the loom-to-memref lowering pass for textual pipelines.
 */
void registerLoomToMemRefLoweringPass();

} // namespace passes
} // namespace loom

