/**
 * @file FuncOpToTTKernel.h
 * @brief Header for function specialization pass that splits functions into
 *        compute, reader, and writer kernels.
 *
 * @details This pass clones each func::FuncOp into three specialized versions:
 *          - `__compute`: retains compute ops (e.g., linalg.matmul), erases
 *                        memory stores
 *          - `__reader` : retains memory *load* ops, erases memory stores and
 *                        compute ops
 *          - `__writer` : retains memory *store* ops, erases memory loads and
 *                        compute ops
 *          This separation happens before MemoryOp/ComputeOp lowering so each
 *          specialized function can be lowered independently, following the
 *          compute/reader/writer split used in the Triton-Tenstorrent flow.
 */

#ifndef LOOM_PASSES_TILELOOMTOTTKERNEL_FUNCOPTOTTKERNEL_H
#define LOOM_PASSES_TILELOOMTOTTKERNEL_FUNCOPTOTTKERNEL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace mlir {
namespace loom {

/**
 * @brief Per-function tracking data for compile-arg indices.
 *
 * @details Each function gets its own tracking state so that compile-arg
 *          indices start from 0 for each specialized function.
 */
struct FunctionTrackingData {
  /// Map from input memref to its compile-arg indices (cbIndex, baseAddrIndex).
  llvm::DenseMap<Value, std::pair<int64_t, int64_t>> inputToData;
  /// Map from output memref to its compile-arg indices (cbIndex, baseAddrIndex).
  llvm::DenseMap<Value, std::pair<int64_t, int64_t>> outputToData;
  /// Map from alloc result to its assigned CB index.
  llvm::DenseMap<Value, int64_t> allocToCbIndex;
  /// Map from index-typed value to its compile-arg index.
  llvm::DenseMap<Value, int64_t> indexToArgIndex;
  /// Next compile-arg index.
  int64_t nextCompileArgIndex = 0;
};

/**
 * @brief Tracks function arguments and assigns compile-arg indices.
 *
 * @details Supports tracking for:
 *          - Memref arguments: assigned pairs of (cbIndex, baseAddrIndex)
 *          - Index arguments: assigned a single compile-arg index
 *          Each function maintains its own independent set of indices.
 */
class CompileArgTracker {
public:
  /**
   * @brief Encapsulates compile-arg indices for a memref.
   */
  struct MemrefData {
    int64_t cbIndex;
    int64_t baseAddrIndex;
  };

  /**
   * @brief Get or create a CB index for the given L1 alloc.
   * @param alloc The `memref.alloc` op annotated with `{loom.alloc ...}`.
   * @return A stable, non-negative compile-arg index for this allocation.
   */
  int64_t getOrCreateAlloc(memref::AllocOp alloc);

  /**
   * @brief Get or create input data for the given input memref.
   * @param inputMemref Input memref value (typically a function argument).
   * @param alloc The L1 alloc associated with this input.
   * @return MemrefData containing CB index and base address index.
   */
  MemrefData getOrCreate(Value inputMemref, memref::AllocOp alloc);

  /**
   * @brief Get or create output data for the given output memref.
   *
   * @details Stores use the output (DRAM) memref to recover a compile-arg index
   *          for the base address. Unlike inputs, the output memref may not be
   *          associated with a `{loom.alloc ...}` L1 allocation, so we assign a
   *          fresh (cbIndex, baseAddrIndex) pair keyed by the output memref
   *          value itself.
   *
   * @param outputMemref Output memref value (typically a function argument).
   * @param op The operation requesting the output data (used to find parent function).
   * @return MemrefData containing a stable CB index and base address index.
   */
  MemrefData getOrCreateOutput(Value outputMemref, Operation *op);

  /**
   * @brief Get or create a compile-arg index for an index-typed value.
   * @param indexValue The index-typed value (typically a function argument).
   * @param op The operation requesting the index (used to find parent function).
   * @return A stable compile-arg index for this value.
   */
  int64_t getOrCreateIndex(Value indexValue, Operation *op);

  /**
   * @brief Get the tracking data for a function, or nullptr if not tracked.
   * @param func The function to look up.
   * @return Pointer to tracking data, or nullptr.
   */
  FunctionTrackingData *getFuncData(func::FuncOp func);

  /**
   * @brief Get or create tracking data for a function.
   * @param func The function to track.
   * @return Pointer to tracking data.
   */
  FunctionTrackingData *getOrCreateFuncData(func::FuncOp func);

private:
  /// Get the parent function of an operation.
  func::FuncOp getParentFunc(Operation *op) const;

  /// Get or create tracking data for the function containing the given operation.
  FunctionTrackingData *getOrCreateFuncData(Operation *op);

  /// Map from function to its tracking data.
  llvm::DenseMap<func::FuncOp, FunctionTrackingData> funcToData;
};

/**
 * @brief Specialize functions into compute and data variants.
 *
 * @details For each func::FuncOp in the module, this creates three clones:
 *          - `<name>__compute`: compute-only kernel (stores erased)
 *          - `<name>__reader` : reader kernel (loads only)
 *          - `<name>__writer` : writer kernel (stores only)
 *          The original function is erased after cloning.
 *
 * @param module The module containing functions to specialize.
 */
void specializeFunctionsForTTKernel(ModuleOp module);

/**
 * @brief Replace function arguments with GetCompileArgValOp operations.
 *
 * @details For each function argument:
 *          - Memref types: replaced with GetCompileArgValOp returning CB type
 *          - Index types: replaced with GetCompileArgValOp returning i32
 *          After replacement, all function arguments are removed.
 *
 * @param func The function to process.
 * @param tracker The compile-arg tracker to use for index assignment.
 * @param typeConverter Type converter for memref-to-CB conversion.
 * @param rewriter The pattern rewriter for IR modifications.
 * @return success if all arguments were replaced, failure otherwise.
 */
LogicalResult replaceFuncArgsWithCompileArgs(
    func::FuncOp func, std::shared_ptr<CompileArgTracker> tracker,
    TypeConverter &typeConverter, OpBuilder &rewriter);

/**
 * @brief Erase all arguments from a function.
 *
 * @details Ensures the entry block has no remaining argument uses and then
 *          drops every argument, updating the function type to have an empty
 *          input list. Returns failure if any argument still has uses.
 *
 * @param func Function to mutate.
 * @return success on erase, failure if any argument is still used.
 */
LogicalResult removeAllFunctionArguments(func::FuncOp func);

} // namespace loom
} // namespace mlir

#endif // LOOM_PASSES_TILELOOMTOTTKERNEL_FUNCOPTOTTKERNEL_H

