/**
 * @file utils.h
 * @brief Common utilities for function cloning and enumeration in passes.
 * @details
 * This utility layer provides reusable functions for cloning and managing
 * function variants during pass enumeration phases. These utilities are
 * used by passes like enumerate_hw_mapping, hoist_block_loading, and
 * enumerate_copy_broadcast to avoid code duplication.
 *
 * Core functionality:
 * - Clone functions with automatic insertion point management
 * - Apply modifications to cloned functions with validation
 * - Collect functions from modules (avoiding iterator invalidation)
 *
 * Intended usage:
 * - Include this header in pass implementations that need to enumerate
 *   function variants
 * - Use cloneAndInsertFunction for simple cloning with renaming
 * - Use cloneModifyAndInsertFunction for cloning with validation logic
 */

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

namespace loom {
namespace utils {

/**
 * @brief Clone a function and insert it into the module with a new name.
 *
 * @details Creates a clone of the original function using IRMapping,
 * sets the new name, and manages the insertion point. If insertAfter is
 * nullptr, uses the builder's current insertion point. After insertion,
 * the builder's insertion point is updated to after the cloned function.
 *
 * @param builder OpBuilder for cloning and insertion operations
 * @param originalFunc Original function to clone
 * @param newName New name for the cloned function
 * @param insertAfter Insert after this operation (nullptr uses current point)
 * @return The cloned function with the new name
 */
mlir::func::FuncOp cloneAndInsertFunction(
    mlir::OpBuilder &builder,
    mlir::func::FuncOp originalFunc,
    llvm::StringRef newName,
    mlir::Operation *insertAfter = nullptr);

/**
 * @brief Clone function, apply modifications, then insert if valid.
 *
 * @details This is a convenience wrapper around cloneAndInsertFunction that
 * applies a modification callback to the cloned function. If the modifier
 * returns success(), the function is kept and returned. If the modifier
 * returns failure(), the cloned function is erased and nullptr is returned.
 * The insertion point is updated only if the function is kept.
 *
 * Typical usage pattern:
 * @code
 * auto cloned = cloneModifyAndInsertFunction(
 *     builder, originalFunc, "new_name",
 *     [&](func::FuncOp func) -> LogicalResult {
 *         // Apply transformations to func
 *         if (transformationFailed)
 *             return failure();
 *         return success();
 *     },
 *     insertAfter);
 * if (cloned) {
 *     // Function was successfully modified and inserted
 * }
 * @endcode
 *
 * @param builder OpBuilder for cloning and insertion operations
 * @param originalFunc Original function to clone
 * @param newName New name for the cloned function
 * @param modifier Callback returning success() to keep, failure() to discard
 * @param insertAfter Insert after this operation (nullptr uses current point)
 * @return Cloned and modified function, or nullptr if modifier returned failure
 */
mlir::func::FuncOp cloneModifyAndInsertFunction(
    mlir::OpBuilder &builder,
    mlir::func::FuncOp originalFunc,
    llvm::StringRef newName,
    std::function<mlir::LogicalResult(mlir::func::FuncOp)> modifier,
    mlir::Operation *insertAfter = nullptr);

/**
 * @brief Collect all functions in a module into a vector.
 *
 * @details Walks the module and collects all func::FuncOp operations into
 * a SmallVector. This is useful for avoiding iterator invalidation when
 * passes need to clone or modify functions while iterating over them.
 *
 * Typical usage:
 * @code
 * auto funcs = collectFunctions(module);
 * for (auto func : funcs) {
 *     // Can safely clone or modify func without iterator issues
 * }
 * @endcode
 *
 * @param module Module to collect functions from
 * @return Vector containing all functions in the module
 */
llvm::SmallVector<mlir::func::FuncOp> collectFunctions(mlir::ModuleOp module);

} // namespace utils
} // namespace loom

