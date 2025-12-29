/**
 * @file materialize.h
 * @brief Materialize pass for canonicalizing IR.
 * @details
 * This pass performs canonicalization on the IR to simplify and normalize
 * operations. It operates on the module level and can be used to clean up
 * and optimize the IR after various transformations.
 *
 * Usage
 * - Register: `loom::passes::registerMaterializePass()`
 * - CLI: `--loom-materialize`
 */

#pragma once

#include "mlir/Pass/Pass.h"

namespace loom {
namespace passes {

/**
 * \brief Create a pass that materializes and canonicalizes IR.
 *
 * This pass performs canonicalization transformations on the module to
 * simplify and normalize operations.
 */
std::unique_ptr<mlir::Pass> createMaterializePass();

/**
 * \brief Register the materialize pass for textual pipelines.
 */
void registerMaterializePass();

} // namespace passes
} // namespace loom

