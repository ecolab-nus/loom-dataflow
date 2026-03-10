/**
 * @file utils.h
 * @brief Common utilities for function cloning and enumeration in passes.
 */

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

namespace loom {
namespace utils {

using SymbolicDim = mlir::OpFoldResult;

/**
 * @brief Get the parent module operation that directly contains a function.
 */
mlir::ModuleOp getParentModule(mlir::func::FuncOp func);

/**
 * @brief Clone a function and insert it into the module with a new name.
 */
mlir::func::FuncOp
cloneAndInsertFunction(mlir::OpBuilder &builder,
                       mlir::func::FuncOp originalFunc, llvm::StringRef newName,
                       mlir::Operation *insertAfter = nullptr);

/**
 * @brief Clone a function with module wrapper and insert it with a new name.
 */
mlir::func::FuncOp cloneAndInsertFunctionWithModuleWrapper(
    mlir::OpBuilder &builder, mlir::func::FuncOp originalFunc,
    llvm::StringRef newName, mlir::DictionaryAttr moduleAttrs,
    mlir::Operation *insertAfter = nullptr);

/**
 * @brief Clone function, apply modifications, then insert if valid.
 */
mlir::func::FuncOp cloneModifyAndInsertFunction(
    mlir::OpBuilder &builder, mlir::func::FuncOp originalFunc,
    llvm::StringRef newName,
    std::function<mlir::LogicalResult(mlir::func::FuncOp)> modifier,
    mlir::Operation *insertAfter = nullptr);

/**
 * @brief Clone function with module wrapper, apply modifications, then insert
 * if valid.
 */
mlir::func::FuncOp cloneModifyAndInsertFunctionWithModuleWrapper(
    mlir::OpBuilder &builder, mlir::func::FuncOp originalFunc,
    llvm::StringRef newName, mlir::DictionaryAttr moduleAttrs,
    std::function<mlir::LogicalResult(mlir::func::FuncOp)> modifier,
    mlir::Operation *insertAfter = nullptr);

/**
 * @brief Collect all functions in a module into a vector.
 */
llvm::SmallVector<mlir::func::FuncOp> collectFunctions(mlir::ModuleOp module);

/**
 * @brief Clone function with module wrapper.
 */
mlir::func::FuncOp cloneFuncWithConstraints(
    mlir::OpBuilder &builder, mlir::func::FuncOp originalFunc,
    llvm::StringRef newName, mlir::DictionaryAttr moduleAttrs,
    llvm::StringRef passName,
    std::function<mlir::LogicalResult(mlir::func::FuncOp)> modifier,
    mlir::Operation *insertAfter = nullptr);

struct AllocInfo {
  llvm::SmallVector<llvm::StringRef> dims;
  int64_t elemSize;
};

/**
 * @brief Collect all loom.alloc operations on @L1 and analyze their dimensions.
 */
llvm::SmallVector<AllocInfo> collectL1AllocInfos(mlir::func::FuncOp func);

/**
 * @brief Trace an SSA value back to a symbolic block size name.
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
