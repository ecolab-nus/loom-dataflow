/**
 * @file FuncOpToTTKernel.h
 * @brief Header for function specialization pass that splits functions into
 *        compute, reader, writer, and host helper kernels.
 *
 * @details This pass clones each func::FuncOp into five specialized versions:
 *          - `__compute`: retains compute ops (e.g., linalg.matmul), erases
 *                        memory stores
 *          - `__reader` : retains memory *load* ops, erases memory stores and
 *                        compute ops
 *          - `__writer` : retains memory *store* ops, erases memory loads and
 *                        compute ops
 *          - `__host_cpp`   : erases all compute/load/store ops and emits
 *                             vector-backed host setup
 *          - `__host_pybind`: erases all compute/load/store ops and emits
 *                             buffer-backed host setup
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
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {
namespace loom {

/**
 * @brief Runtime tuple values created for a DRAM/L1 binding.
 *
 * @details Stores the CB value, base address value, tensor accessor, and
 *          multicast helper values materialized from `ttkernel.get_arg_val`.
 *          This data is shared by per-copy bindings and any internal CB users
 *          that need the same runtime tuple shape.
 */
struct MemrefArgData {
  /// The CB value created for this memref argument.
  Value cb;
  /// The base address value created for this memref argument.
  Value baseAddr;
  /// The tensor accessor value created for this memref argument.
  Value tensorAccessor;
  /// The start x NOC coordinate of the multicast destination range.
  Value mcast_dest_noc_start_x;
  /// The start y NOC coordinate of the multicast destination range.
  Value mcast_dest_noc_start_y;
  /// The end x NOC coordinate of the multicast destination range.
  Value mcast_dest_noc_end_x;
  /// The end y NOC coordinate of the multicast destination range.
  Value mcast_dest_noc_end_y;
  /// The number of destinations for multicast.
  Value mcast_dest_num;
  /// The sender x NOC coordinate for multicast.
  Value mcast_sender_noc_x;
  /// The sender y NOC coordinate for multicast.
  Value mcast_sender_noc_y;
  /// The sender semaphore address for multicast.
  Value mcast_sender_semaphore_addr;
  /// The receiver semaphore address for multicast.
  Value mcast_receiver_semaphore_addr;
  /// The L1 multicast sender semaphore address pointer.
  Value mcast_sender_semaphore_addr_ptr;
  /// The L1 multicast receiver semaphore address pointer.
  Value mcast_receiver_semaphore_addr_ptr;
  /// The noc address of sender semaphore
  Value mcast_sender_semaphore_noc_addr;
  /// The noc address of receiver semaphore
  Value mcast_receiver_semaphore_noc_addr;
  /// The NOC ID resolved from the owning data-movement kernel processor.
  int8_t noc_id = 0;
  /// The number of tiles for this memref (stored as Value for use in CB ops).
  Value num_tiles;
  /// The location attribute where this memref argument data was initialized.
  LocationAttr initLoc;
};

/**
 * @brief Runtime tuple plus source metadata for a specific DRAM/L1 `loom.copy`.
 *
 * @details Each DRAM/L1 copy site owns an independent binding slot even when
 *          the underlying DRAM memref argument is shared by multiple copies.
 */
struct CopyBindingData : public MemrefArgData {
  /// Stable binding slot shared across specialized kernels.
  int64_t slot = -1;
  /// Original memref argument on the DRAM side of the copy.
  Value linkedMemrefArg;
  /// Non-DRAM endpoint of the copy (typically semaphore_take or alloc).
  Value l1Endpoint;
  /// True when the binding is DRAM -> L1.
  bool isLoad = false;
  /// True when the binding is L1 -> DRAM.
  bool isStore = false;
};
/**
 * @brief Data created for an index input argument.
 *
 * @details Stores the compile-arg value created by GetArgValOp
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
 *          - Memref arguments: creates a CB and base address via GetArgValOp
 *          - Index arguments: creates a single compile-arg value via GetArgValOp
 *          The created values are stored in maps for later usage.
 */
class CompileArgTracker {
public:
  struct ReduceRuntimeArgs {
    Value readySemaphore;
    Value tokenSemaphore;
    Value tokenSemaphoreMcastDestStartX;
    Value tokenSemaphoreMcastDestStartY;
    Value tokenSemaphoreMcastDestEndX;
    Value tokenSemaphoreMcastDestEndY;
  };
  /**
   * @brief Process all input arguments of a function and create compile-arg values.
   *
   * @details For each function argument:
   *          - Memref types: creates two GetArgValOp (CB and base address)
   *          - Index types: creates one GetArgValOp and casts to index
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
   * @details Allocates a new compile-arg index and creates a GetArgValOp
   *          followed by an index cast. Stores the result for later retrieval.
   *          This is used for loop induction variables that are not function args.
   *
   * @param value The value to create a compile-arg for.
   * @param loc The location for created ops.
   * @param rewriter The builder for IR modifications.
   * @return The index-casted value representing the compile-arg.
   */
  Value createIndexCompileArg(Value value, Location loc, OpBuilder &rewriter);

  /**
   * @brief Create a typed compile-arg value for non-argument lowering sites.
   *
   * @details Allocates the next per-function compile-arg index and emits a
   *          `GetArgValOp` with `resultType`. This is used by conversion
   *          patterns that need an internal kernel argument (e.g., CB handles
   *          for `loom.alloc`) that is not directly tied to an original
   *          function block argument.
   *
   * @param loc Location for the created operations.
   * @param rewriter Builder used to create operations.
   * @param funcOp Parent function that scopes compile-arg index allocation.
   * @param resultType Result type for the `GetArgValOp`.
   * @return The created compile-arg value, or null on invalid input.
   */
  Value createTypedCompileArg(Location loc, OpBuilder &rewriter,
                              Operation *funcOp, Type resultType);

  /**
   * @brief Create or reuse a typed compile-arg at a fixed absolute arg index.
   *
   * @details This bypasses sequential index allocation and is used when an
   *          operation must bind to a deterministic runtime-arg slot shared
   *          across specialized kernels.
   *
   * @param loc Location for the created operations.
   * @param rewriter Builder used to create operations.
   * @param funcOp Parent function that scopes compile-arg allocation.
   * @param argIndex Absolute runtime-arg index to materialize.
   * @param resultType Result type for the `GetArgValOp`.
   * @return The created/reused compile-arg value, or null on invalid input.
   */
  Value createTypedCompileArgAtIndex(Location loc, OpBuilder &rewriter,
                                     Operation *funcOp, int64_t argIndex,
                                     Type resultType);

  /**
   * @brief Get the precomputed runtime-arg base index for internal CB handles.
   *
   * @details Internal CB slots are emitted after memref/reduce/scalar args and
   *          core-coordinate args. This returns the per-function base index
   *          used to translate a stable internal slot id to an absolute
   *          runtime-arg index.
   */
  int64_t getInternalCbBaseArgIndex(Operation *funcOp) const;

  /// Get function-level reduce runtime args if available.
  const ReduceRuntimeArgs *getReduceRuntimeArgs(Operation *funcOp) const;
  ReduceRuntimeArgs *getReduceRuntimeArgs(Operation *funcOp);

  /// Get a per-function scalar runtime arg (packed scalar word) by site id.
  Value getScalarRuntimeArg(Operation *funcOp, int64_t siteId) const;

  /**
   * @brief Get the runtime tuple data for a per-copy binding slot.
   *
   * @param funcOp Parent function that owns the binding.
   * @param slot Stable copy-binding slot.
   * @return Pointer to CopyBindingData if found, nullptr otherwise.
   */
  CopyBindingData *getCopyBindingData(Operation *funcOp, int64_t slot);

  /**
   * @brief Get the number of per-copy runtime bindings in a function.
   *
   * @param funcOp Parent function.
   * @return Number of annotated DRAM/L1 copy bindings.
   */
  int64_t getCopyBindingCount(Operation *funcOp) const;

  /**
   * @brief Append a value to the tracker core list.
   *
   * @details The core list is a simple ordered collection of values that
   *          represent per-kernel coordinates/IDs (e.g., scf.parallel IVs)
   *          materialized as compile-time arguments. This enables other
   *          lowering patterns to query the set of core-coordinate values that
   *          were created while lowering a kernel.
   *
   * @param value The value to append to the core list.
   */
  void appendToCoreList(Operation *funcOp, Value value);

  /**
   * @brief Get the current core list.
   *
   * @return A read-only view of the core list values in insertion order.
   */
  ArrayRef<Value> getCoreList(Operation *funcOp) const;

  /**
   * @brief Clear the current core list.
   */
  void clearCoreList(Operation *funcOp);

  /**
   * @brief Record a per-function core coordinate value for a mapped dim.
   *
   * @param funcOp Parent function operation.
   * @param dimName Spatial dim name ("x" or "y", case-insensitive).
   * @param value Core coordinate value associated with the dim.
   */
  void setCoreCoordForDim(Operation *funcOp, StringRef dimName, Value value);

  /**
   * @brief Lookup a per-function core coordinate value for a mapped dim.
   *
   * @param funcOp Parent function operation.
   * @param dimName Spatial dim name ("x" or "y", case-insensitive).
   * @return The recorded value, or nullptr if unavailable.
   */
  Value getCoreCoordForDim(Operation *funcOp, StringRef dimName) const;

  /**
   * @brief Get the next unique index for TensorAccessorArgs.
   *
   * @details Allocates and returns a new unique index for use with
   *          TensorAccessorArgsOp. Each tensor accessor should have different
   *          cta_base and crta_base indices to distinguish them.
   *
   * @param funcOp The function operation to scope the index allocation.
   * @return The next available tensor accessor args index.
   */
  int64_t getNextTensorAccessorArgsIndex(Operation *funcOp);

private:
  // Helper to get and increment the compile-arg index for a specific function.
  int64_t getAndIncrementIndex(Operation *funcOp);

  // Helper to materialize GetArgValOp at an explicit absolute arg index.
  Value createGetArgValOpAtIndex(Location loc, OpBuilder &rewriter,
                                 Operation *funcOp, int64_t argIndex,
                                 Type resultType);

  /**
   * @brief Create a GetArgValOp with an automatically generated index.
   *
   * @details Generates a new compile-arg index for the given function operation,
   *          creates a constant i32 index value, and uses it to create a GetArgValOp
   *          with the specified result type. This is a common pattern used throughout
   *          the codebase for creating compile-arg operations.
   *
   * @param loc Location for the created operations.
   * @param rewriter Builder for creating operations.
   * @param funcOp The function operation to generate the index for.
   * @param resultType The result type for the GetArgValOp.
   * @return The result Value from the GetArgValOp.
   */
  Value createGetArgValOp(Location loc, OpBuilder &rewriter, Operation *funcOp,
                          Type resultType);

  // Map from FuncOp (as Operation*) to the next available compile-arg index.
  llvm::DenseMap<Operation *, int64_t> funcToNextArgIndex;

  // Per-function base runtime-arg index where internal CB slots begin.
  llvm::DenseMap<Operation *, int64_t> funcToInternalCbBaseArgIndex;

  // Per-function cache for explicit-index compile args.
  llvm::DenseMap<Operation *, llvm::DenseMap<int64_t, Value>>
      funcToExplicitCompileArgs;

  // Map from FuncOp (as Operation*) to the next available tensor accessor args index.
  llvm::DenseMap<Operation *, int64_t> funcToNextTensorAccessorArgsIndex;

  /// Map from memref argument to its created CB and base address.
  llvm::DenseMap<Value, MemrefArgData> memrefArgToData;

  /// Map from memref argument to its shared TensorAccessorArgs index.
  llvm::DenseMap<Value, int64_t> memrefArgToTensorAccessorArgsIndex;

  /// Per-function runtime tuple data keyed by stable copy-binding slot.
  llvm::DenseMap<Operation *, llvm::DenseMap<int64_t, CopyBindingData>>
      funcToCopyBindingData;
  
  /// Map from index argument to its created compile-arg value.
  llvm::DenseMap<Value, IndexArgData> indexArgToData;

  /// Per-function ordered list of "core" values (e.g., compile-arg-based core coordinates).
  llvm::DenseMap<Operation *, llvm::SmallVector<Value, 2>> funcToCoreList;

  struct CoreCoordByDim {
    Value x;
    Value y;
  };

  /// Per-function explicit core coordinate values by mapped dim name.
  llvm::DenseMap<Operation *, CoreCoordByDim> funcToCoreCoordsByDim;

  /// Optional per-function reduce synchronization runtime args.
  llvm::DenseMap<Operation *, ReduceRuntimeArgs> funcToReduceRuntimeArgs;

  /// Per-function scalar runtime args in scalar-site order.
  llvm::DenseMap<Operation *, llvm::SmallVector<Value, 4>>
      funcToScalarRuntimeArgs;
};

/**
 * @brief Mark matmul functions for merged B-reader/writer specialization.
 *
 * @details Identifies supported `linalg.matmul` functions before cloning and
 *          tags the relevant `loom.copy` ops so specialization can keep the
 *          A-side load in `__reader` and move the B-side load into
 *          `__writer`.
 *
 * @param module The module containing original (unspecialized) functions.
 * @return success if all eligible matmul functions were annotated, failure on
 *         malformed supported input.
 */
LogicalResult prepareMatmulBReaderMerge(ModuleOp module);

/**
 * @brief Annotate DRAM vector-load copy sites with vec->tile metadata.
 *
 * @details Scans load-direction `loom.copy` ops whose source is
 *          `memref.reinterpret_cast` and whose destination is rank-1 static
 *          memref storage. For each such load, it inspects downstream
 *          `linalg.generic` usage and records:
 *          - `loom.ttkernel.vec_kind`  : `none` | `row_bcast` | `col_bcast`
 *          - `loom.ttkernel.vec_tiles` : required tile pages in reader CB
 *
 *          Conflicting usage intents for the same load site are rejected.
 *
 * @param module The module containing unspecialized functions.
 * @return success on valid annotation, failure on conflicting/unsupported use.
 */
LogicalResult annotateVecLoadUsage(ModuleOp module);

/**
 * @brief Specialize functions into compute, data, and host variants.
 *
 * @details For each func::FuncOp in the module, this creates five clones:
 *          - `<name>__compute`: compute-only kernel (stores erased)
 *          - `<name>__reader` : reader kernel (loads only)
 *          - `<name>__writer` : writer kernel (stores only)
 *          - `<name>__host_cpp`    : vector-backed host helper
 *          - `<name>__host_pybind` : buffer-backed host helper
 *          The original function is erased after cloning.
 *
 * @param module The module containing functions to specialize.
 */
void specializeFunctionsForTTKernel(ModuleOp module);

/**
 * @brief Replace function arguments with GetArgValOp operations.
 *
 * @details For each function argument:
 *          - Memref types: replaced with GetArgValOp returning CB type
 *          - Index types: replaced with GetArgValOp returning i32
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
