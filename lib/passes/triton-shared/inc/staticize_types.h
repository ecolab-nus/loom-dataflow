/**
 * @file staticize_types.h
 * @brief Pass to convert dynamic memref/tensor types to static types.
 * @details
 * This pass converts operations with dynamic shapes to use static shapes
 * when the dynamic sizes can be resolved to constants. It handles:
 * - memref.alloc: Converts dynamic allocations to static when sizes are constants
 * - tensor.empty: Converts dynamic tensor creation to static when sizes are constants
 *
 * This pass should run after MaterializePass which converts symbolic block
 * sizes to arith.constant, so that the constant sizes can be detected and
 * used to staticize types.
 *
 * Usage
 * - Register: `loom::passes::registerStaticizeTypesPass()`
 * - CLI: `--loom-staticize-types`
 */

#pragma once

#include "mlir/Pass/Pass.h"

namespace loom {
namespace passes {

/**
 * @brief Create a pass that staticizes memref and tensor types.
 *
 * This pass converts dynamic allocations and tensor creations to use
 * static shapes when all dynamic sizes are arith.constant operations.
 */
std::unique_ptr<mlir::Pass> createStaticizeTypesPass();

/**
 * @brief Register the staticize types pass for textual pipelines.
 */
void registerStaticizeTypesPass();

} // namespace passes
} // namespace loom

