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

// SymbolicDim and traceShape live in trace_shape.h; re-exported here so that
// existing consumers of utils.h pick up both transparently.
#include "trace_shape.h"

namespace loom {
namespace utils {

/**
 * @brief Get the parent module operation that directly contains a function.
 */
mlir::ModuleOp getParentModule(mlir::func::FuncOp func);

/**
 * @brief Collect all functions in a module into a vector.
 */
llvm::SmallVector<mlir::func::FuncOp> collectFunctions(mlir::ModuleOp module);

/**
 * @brief Clone a function and insert it into the IR.
 * @details Optionally wraps the clone in a fresh module carrying
 * `moduleAttrs`, runs `modifier` on the clone before insertion, and on a
 * `failure()` from the modifier rolls back by erasing the partial clone.
 * Returns the cloned func, or null if the modifier signaled failure.
 */
mlir::func::FuncOp cloneFunc(
    mlir::OpBuilder &builder, mlir::func::FuncOp originalFunc,
    llvm::StringRef newName,
    mlir::DictionaryAttr moduleAttrs = nullptr,
    std::function<mlir::LogicalResult(mlir::func::FuncOp)> modifier = nullptr,
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

} // namespace utils
} // namespace loom
