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
 * @brief Data created for a memref input argument.
 *
 * @details Stores the CB value and base address value created by 
 *          GetCompileArgValOp for a memref function argument.
 */
struct MemrefArgData {
  /// The CB value created for this memref argument.
  Value cb;
  /// The base address value created for this memref argument.
  Value baseAddr;
  /// The tensor accessor value created for this memref argument.
  Value tensorAccessor;
};

/**
 * @brief Data created for an index input argument.
 *
 * @details Stores the compile-arg value created by GetCompileArgValOp
 *          for an index function argument.
 */
struct IndexArgData {
  /// The compile-arg value (i32) created for this index argument.
  Value compileArg;
  /// The index-casted value for use in operations expecting index type.
  Value indexValue;
};

/**
 * @brief Tracks input arguments and creates compile-arg values for them.
 *
 * @details This simplified tracker focuses only on input arguments:
 *          - Memref arguments: creates a CB and base address via GetCompileArgValOp
 *          - Index arguments: creates a single compile-arg value via GetCompileArgValOp
 *          The created values are stored in maps for later usage.
 */
class CompileArgTracker {
public:
  /**
   * @brief Process all input arguments of a function and create compile-arg values.
   *
   * @details For each function argument:
   *          - Memref types: creates two GetCompileArgValOp (CB and base address)
   *          - Index types: creates one GetCompileArgValOp and casts to index
   *          All created values are stored in internal maps and can be retrieved later.
   *
   * @param func The function to process.
   * @param typeConverter Type converter for memref-to-CB conversion.
   * @param rewriter The builder for IR modifications.
   * @return success if all arguments were processed, failure otherwise.
   */
  LogicalResult processInputArgs(func::FuncOp func, TypeConverter &typeConverter,
                                  OpBuilder &rewriter);

  /**
   * @brief Get the created data for a memref argument.
   * @param arg The memref argument (BlockArgument).
   * @return Pointer to MemrefArgData if found, nullptr otherwise.
   */
  MemrefArgData *getMemrefData(Value arg);

  /**
   * @brief Get the created data for an index argument.
   * @param arg The index argument (BlockArgument).
   * @return Pointer to IndexArgData if found, nullptr otherwise.
   */
  IndexArgData *getIndexData(Value arg);

  /**
   * @brief Get the CB value for a memref argument.
   * @param arg The memref argument (BlockArgument).
   * @return The CB value if found, nullptr otherwise.
   */
  Value getCB(Value arg);

  /**
   * @brief Get the base address value for a memref argument.
   * @param arg The memref argument (BlockArgument).
   * @return The base address value if found, nullptr otherwise.
   */
  Value getBaseAddr(Value arg);

  /**
   * @brief Get the tensor accessor value for a memref argument.
   * @param arg The memref argument (BlockArgument).
   * @return The tensor accessor value if found, nullptr otherwise.
   */
  Value getTensorAccessor(Value arg);

  /**
   * @brief Get the index value for an index argument.
   * @param arg The index argument (BlockArgument).
   * @return The index value if found, nullptr otherwise.
   */
  Value getIndexValue(Value arg);



  /**
   * @brief Create a compile-arg for an index-typed value (e.g., loop IV).
   *
   * @details Allocates a new compile-arg index and creates a GetCompileArgValOp
   *          followed by an index cast. Stores the result for later retrieval.
   *          This is used for loop induction variables that are not function args.
   *
   * @param value The value to create a compile-arg for.
   * @param loc The location for created ops.
   * @param rewriter The builder for IR modifications.
   * @return The index-casted value representing the compile-arg.
   */
  Value createIndexCompileArg(Value value, Location loc, OpBuilder &rewriter);

private:
  // Helper to get and increment the compile-arg index for a specific function.
  int64_t getAndIncrementIndex(Operation *funcOp);

  // Map from FuncOp (as Operation*) to the next available compile-arg index.
  llvm::DenseMap<Operation *, int64_t> funcToNextArgIndex;

  /// Map from memref argument to its created CB and base address.
  llvm::DenseMap<Value, MemrefArgData> memrefArgToData;
  
  /// Map from index argument to its created compile-arg value.
  llvm::DenseMap<Value, IndexArgData> indexArgToData;
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

