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

// Forward declarations for Loom operations
namespace loom {
class ConstraintSpaceOp;
} // namespace loom

namespace loom {

// Forward declarations for Loom operations
class ConstraintSpaceOp;

namespace utils {

using SymbolicDim = mlir::OpFoldResult;

/**
 * @brief Get the parent module operation that directly contains a function.
 *
 * @details Returns the immediate parent ModuleOp of a function. This is useful
 * for extracting attributes from the wrapper module in nested module
 * structures.
 *
 * @param func Function to get the parent module for
 * @return The parent ModuleOp, or nullptr if the function is not in a module
 */
mlir::ModuleOp getParentModule(mlir::func::FuncOp func);

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
mlir::func::FuncOp
cloneAndInsertFunction(mlir::OpBuilder &builder,
                       mlir::func::FuncOp originalFunc, llvm::StringRef newName,
                       mlir::Operation *insertAfter = nullptr);

/**
 * @brief Clone a function with module wrapper and insert it with a new name.
 *
 * @details Creates a wrapper ModuleOp with the specified attributes, clones
 * the original function into it, sets the new name, and manages the insertion
 * point. The wrapper module is inserted at the specified location.
 *
 * @param builder OpBuilder for cloning and insertion operations
 * @param originalFunc Original function to clone
 * @param newName New name for the cloned function
 * @param moduleAttrs Attributes to set on the wrapper module
 * @param insertAfter Insert after this operation (nullptr uses current point)
 * @return The cloned function with the new name (inside its wrapper module)
 */
mlir::func::FuncOp cloneAndInsertFunctionWithModuleWrapper(
    mlir::OpBuilder &builder, mlir::func::FuncOp originalFunc,
    llvm::StringRef newName, mlir::DictionaryAttr moduleAttrs,
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
    mlir::OpBuilder &builder, mlir::func::FuncOp originalFunc,
    llvm::StringRef newName,
    std::function<mlir::LogicalResult(mlir::func::FuncOp)> modifier,
    mlir::Operation *insertAfter = nullptr);

/**
 * @brief Clone function with module wrapper, apply modifications, then insert
 * if valid.
 *
 * @details Creates a wrapper ModuleOp with the specified attributes, clones
 * the function into it, applies the modifier callback, and manages insertion.
 * If the modifier returns failure(), both the function and wrapper module are
 * erased and nullptr is returned.
 *
 * @param builder OpBuilder for cloning and insertion operations
 * @param originalFunc Original function to clone
 * @param newName New name for the cloned function
 * @param moduleAttrs Attributes to set on the wrapper module
 * @param modifier Callback returning success() to keep, failure() to discard
 * @param insertAfter Insert after this operation (nullptr uses current point)
 * @return Cloned and modified function, or nullptr if modifier returned failure
 */
mlir::func::FuncOp cloneModifyAndInsertFunctionWithModuleWrapper(
    mlir::OpBuilder &builder, mlir::func::FuncOp originalFunc,
    llvm::StringRef newName, mlir::DictionaryAttr moduleAttrs,
    std::function<mlir::LogicalResult(mlir::func::FuncOp)> modifier,
    mlir::Operation *insertAfter = nullptr);

/**
 * @brief Collect all functions in a module into a vector.
 *
 * @details Recursively walks the module (including nested modules) and collects
 * all func::FuncOp operations into a SmallVector. This is useful for avoiding
 * iterator invalidation when passes need to clone or modify functions while
 * iterating over them. Works with both flat and nested module structures.
 *
 * Typical usage:
 * @code
 * auto funcs = collectFunctions(module);
 * for (auto func : funcs) {
 *     // Can safely clone or modify func without iterator issues
 * }
 * @endcode
 *
 * @param module Module to collect functions from (recursively searches nested
 * modules)
 * @return Vector containing all functions in the module and its nested modules
 */
llvm::SmallVector<mlir::func::FuncOp> collectFunctions(mlir::ModuleOp module);

/**
 * @brief Clone function with module wrapper including deep-cloned constraint
 * space.
 *
 * @details Creates a wrapper ModuleOp with the specified attributes, clones the
 * function and constraint space into it, applies the modifier callback, and
 * checks feasibility. If the modified constraint space is infeasible (empty
 * solution set), the wrapper module is erased and nullptr is returned.
 *
 * This is the primary entry point for constraint-aware function variant
 * generation. The modifier callback receives both the cloned function and the
 * cloned constraint space, allowing it to add new constraints based on
 * pass-specific logic.
 *
 * Typical usage:
 * @code
 * auto cloned = cloneFuncWithConstraints(
 *     builder, originalFunc, "new_name", moduleAttrs, "MyPass",
 *     [&](func::FuncOp func, loom::ConstraintSpaceOp csOp) -> LogicalResult {
 *         // Add new constraints based on pass logic
 *         loom::lcs::addLinearConstraint(csOp, {"M", "N"}, constraintMap);
 *         return success();
 *     },
 *     insertAfter);
 * if (cloned) {
 *     // Function was valid and inserted
 * }
 * @endcode
 *
 * @param builder OpBuilder for cloning and insertion operations
 * @param originalFunc Original function to clone
 * @param newName New name for the cloned function
 * @param moduleAttrs Attributes to set on the wrapper module
 * @param passName Name of the pass for provenance tracking
 * @param modifier Callback that can modify the func and add constraints to csOp
 * @param insertAfter Insert after this operation (nullptr uses current point)
 * @return Cloned function if feasible, nullptr if infeasible or modifier failed
 */
mlir::func::FuncOp cloneFuncWithConstraints(
    mlir::OpBuilder &builder, mlir::func::FuncOp originalFunc,
    llvm::StringRef newName, mlir::DictionaryAttr moduleAttrs,
    llvm::StringRef passName,
    std::function<mlir::LogicalResult(mlir::func::FuncOp,
                                      loom::ConstraintSpaceOp)>
        modifier,
    mlir::Operation *insertAfter = nullptr);

struct AllocInfo {
  llvm::SmallVector<llvm::StringRef> dims;
  int64_t elemSize;
};

/**
 * @brief Collect all loom.alloc operations on @L1 and analyze their dimensions.
 *
 * @details This function walks the function to find all loom.alloc operations
 * that are allocated on @L1. For each such operation, it traces its dynamic
 * dimensions back to symbolic variable names.
 *
 * @param func The function to analyze.
 * @return Vector of AllocInfo for each L1 allocation.
 */
llvm::SmallVector<AllocInfo> collectL1AllocInfos(mlir::func::FuncOp func);

/**
 * @brief Trace an SSA value back to a symbolic block size name.
 *
 * @param val The SSA value to trace.
 * @return The symbolic variable name if found, empty string otherwise.
 */
llvm::StringRef traceToSymbolicVar(mlir::Value val);

/**
 * @brief Compose and canonicalize all affine.apply operations in a function.
 */
void composeAndCanonicalizeAffineApplies(mlir::func::FuncOp func);

/**
 * @brief Trace an SSA value back to its symbolic shapes.
 */
llvm::SmallVector<SymbolicDim, 4> traceShape(mlir::Value v);

} // namespace utils
} // namespace loom
