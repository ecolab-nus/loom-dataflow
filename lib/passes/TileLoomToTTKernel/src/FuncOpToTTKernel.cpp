/**
 * @file FuncOpToTTKernel.cpp
 * @brief Implementation for function specialization pass.
 *
 * @details This pass clones each func::FuncOp into three specialized versions:
 *          - `__compute`: retains compute ops, erases store operations
 *          - `__reader` : retains memory load ops, erases stores & compute ops
 *          - `__writer` : retains memory store ops, erases loads & compute ops
 *          This loosely mimics the CoreSpecialize pattern from
 *          triton-tenstorrent while remaining TileLoom-specific.
 */

#include "FuncOpToTTKernel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

// TTKernel thread type attribute and enum.
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

using namespace mlir;
using namespace tt::ttkernel;

//===----------------------------------------------------------------------===//
// CompileArgTracker Implementation
//===----------------------------------------------------------------------===//

LogicalResult loom::CompileArgTracker::processInputArgs(
    func::FuncOp func, TypeConverter &typeConverter, OpBuilder &rewriter) {
  Block &entry = func.front();
  Location loc = func.getLoc();

  // Save insertion point and set to start of function body.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&entry);

  // Process each function argument.
  for (BlockArgument arg : entry.getArguments()) {
    Type argType = arg.getType();

    if (isa<MemRefType, UnrankedMemRefType>(argType)) {
      // Memref type: create CB and base address.
      // CB uses nextCompileArgIndex, base address uses nextCompileArgIndex + 1.
      // Memref type: create CB and base address.
      // CB uses nextCompileArgIndex, base address uses nextCompileArgIndex + 1.
      int64_t cbIndex = getAndIncrementIndex(func);
      int64_t baseAddrIndex = getAndIncrementIndex(func);

      // Create GetCompileArgValOp for CB.
      auto cbIdxAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(cbIndex));
      auto cbType = typeConverter.convertType(argType);
      if (!cbType)
        return func.emitError() << "failed to convert memref type to CB type";
      auto cbOp = rewriter.create<GetCompileArgValOp>(loc, cbType, cbIdxAttr);

      // Create GetCompileArgValOp for base address.
      auto baseAddrIdxAttr =
          rewriter.getI32IntegerAttr(static_cast<int32_t>(baseAddrIndex));
      auto baseAddrOp = rewriter.create<GetCompileArgValOp>(
          loc, rewriter.getI32Type(), baseAddrIdxAttr);

      // Create TensorAccessArgs and TensorAccess for base address.
      auto pagesize = GetTileSizeOp::create(rewriter, loc, cbOp.getResult());
      auto baseAddrArgs = rewriter.create<TensorAccessorArgsOp>(loc, baseAddrOp.getResult(), baseAddrOp.getResult());
      auto baseAddrTensorAccess = rewriter.create<TensorAccessorOp>(loc, baseAddrArgs.getResult(), baseAddrOp.getResult(), pagesize);

      // Store the created values keyed by the memref argument.
      // We do NOT replace uses here - the memref argument is used by
      // memref.reinterpret_cast ops which need the original memref type.
      // The memory conversion patterns will use getBaseAddr() to find the
      // pre-created base address when processing load/store ops.
      memrefArgToData[arg] = MemrefArgData{cbOp.getResult(), baseAddrOp.getResult()};

    } else if (argType.isIndex()) {
      // Index type: create a single compile-arg.
      // Index type: create a single compile-arg.
      int64_t argIndex = getAndIncrementIndex(func);

      auto idxAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(argIndex));
      auto compileArgOp =
          rewriter.create<GetCompileArgValOp>(loc, rewriter.getI32Type(), idxAttr);

      // Cast i32 to index for compatibility.
      auto indexCast = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), compileArgOp.getResult());

      // Store the created values.
      indexArgToData[arg] =
          IndexArgData{compileArgOp.getResult(), indexCast.getResult()};

      // Replace uses of the index argument with the casted value.
      arg.replaceAllUsesWith(indexCast.getResult());

    } else {
      // Other types: not supported yet.
      return func.emitError()
             << "unsupported argument type: " << argType;
    }
  }

  return success();
}

loom::MemrefArgData *loom::CompileArgTracker::getMemrefData(Value arg) {
  auto it = memrefArgToData.find(arg);
  if (it != memrefArgToData.end())
    return &it->second;
  return nullptr;
}

loom::IndexArgData *loom::CompileArgTracker::getIndexData(Value arg) {
  auto it = indexArgToData.find(arg);
  if (it != indexArgToData.end())
    return &it->second;
  return nullptr;
}

Value loom::CompileArgTracker::getCB(Value arg) {
  if (auto *data = getMemrefData(arg))
    return data->cb;
  return nullptr;
}

Value loom::CompileArgTracker::getBaseAddr(Value arg) {
  if (auto *data = getMemrefData(arg))
    return data->baseAddr;
  return nullptr;
}

Value loom::CompileArgTracker::getIndexValue(Value arg) {
  if (auto *data = getIndexData(arg))
    return data->indexValue;
  return nullptr;
}

Value loom::CompileArgTracker::createIndexCompileArg(Value value, Location loc,
                                                      OpBuilder &rewriter) {
  // Check if already created.
  if (auto *data = getIndexData(value))
    return data->indexValue;

  // Allocate a new compile-arg index.
  // Find parent function to get the correct index counter.
  Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
  auto funcOp = dyn_cast<func::FuncOp>(parentOp);
  if (!funcOp)
    funcOp = parentOp->getParentOfType<func::FuncOp>();

  if (!funcOp) {
    // Should not happen in valid IR within a function.
    return nullptr;
  }

  // Allocate a new compile-arg index.
  int64_t argIndex = getAndIncrementIndex(funcOp);

  auto idxAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(argIndex));
  auto compileArgOp =
      rewriter.create<GetCompileArgValOp>(loc, rewriter.getI32Type(), idxAttr);

  // Cast i32 to index for compatibility.
  auto indexCast = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), compileArgOp.getResult());

  // Store the created values.
  indexArgToData[value] =
      IndexArgData{compileArgOp.getResult(), indexCast.getResult()};

  return indexCast.getResult();
}

int64_t loom::CompileArgTracker::getAndIncrementIndex(Operation *funcOp) {
  return funcToNextArgIndex[funcOp]++;
}

//===----------------------------------------------------------------------===//
// replaceFuncArgsWithCompileArgs Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::loom::replaceFuncArgsWithCompileArgs(
    func::FuncOp func, std::shared_ptr<CompileArgTracker> tracker,
    TypeConverter &typeConverter, OpBuilder &rewriter) {
  // Delegate to the tracker's processInputArgs method.
  return tracker->processInputArgs(func, typeConverter, rewriter);
}

namespace {

/**
 * @brief Check if a memref.copy is a store operation.
 *
 * @details A store is identified when the target is a reinterpret_cast,
 *          indicating data flows from L1/CB to external DRAM.
 *
 * @param op The memref.copy operation to check.
 * @return true if this is a store operation, false otherwise.
 */
static bool isStoreOp(memref::CopyOp op) {
  return op.getTarget().getDefiningOp<memref::ReinterpretCastOp>() != nullptr;
}

/**
 * @brief Check if a memref.copy is a load operation.
 *
 * @details A load is identified when the source is a reinterpret_cast,
 *          indicating data flows from external DRAM to L1/CB.
 *
 * @param op The memref.copy operation to check.
 * @return true if this is a load operation, false otherwise.
 */
static bool isLoadOp(memref::CopyOp op) {
  return op.getSource().getDefiningOp<memref::ReinterpretCastOp>() != nullptr;
}

/**
 * @brief Check if an operation is a compute operation.
 *
 * @details Currently identifies linalg.matmul as the primary compute op.
 *          Can be extended for other compute operations.
 *
 * @param op The operation to check.
 * @return true if this is a compute operation, false otherwise.
 */
static bool isComputeOp(Operation *op) {
  return isa<linalg::MatmulOp>(op);
}

/**
 * @brief Specialize a function for compute-only execution.
 *
 * @details Clones the function with `__compute` suffix and erases all
 *          store operations (memref.copy where target is reinterpret_cast).
 *          Loads are kept because compute kernels need to read data from CBs.
 *
 * @param func The original function to specialize.
 * @return The specialized compute function.
 */
static func::FuncOp makeComputeFunc(func::FuncOp func) {
  IRMapping mapping;
  auto computeFunc = cast<func::FuncOp>(func->clone(mapping));
  computeFunc.setName((func.getName() + "__compute").str());

  // Collect and erase store operations (keep loads for reading from CBs)
  SmallVector<Operation *, 8> opsToErase;
  computeFunc.walk([&](memref::CopyOp copyOp) {
    if (isStoreOp(copyOp)) {
      opsToErase.push_back(copyOp);
    }
  });

  // Also erase reinterpret_cast ops that were targets of erased stores
  // (they become unused after store removal)
  computeFunc.walk([&](memref::ReinterpretCastOp rcOp) {
    // Check if this reinterpret_cast was the target of a store
    // by seeing if all uses are in the opsToErase list
    bool allUsesErased = true;
    for (OpOperand &use : rcOp.getResult().getUses()) {
      if (auto copyOp = dyn_cast<memref::CopyOp>(use.getOwner())) {
        if (!isStoreOp(copyOp)) {
          allUsesErased = false;
          break;
        }
      } else {
        allUsesErased = false;
        break;
      }
    }
    if (allUsesErased && !rcOp.getResult().getUses().empty()) {
      opsToErase.push_back(rcOp);
    }
  });

  for (Operation *op : opsToErase) {
    op->erase();
  }

  return computeFunc;
}

/**
 * @brief Specialize a function for reader-only data movement.
 *
 * @details Clones the function with `__reader` suffix and erases all
 *          compute operations (linalg.matmul etc.) and store operations
 *          (memref.copy where target is reinterpret_cast). Loads are
 *          retained to model DRAM -> L1 traffic.
 *
 * @param func The original function to specialize.
 * @return The specialized reader function.
 */
static func::FuncOp makeReaderFunc(func::FuncOp func) {
  IRMapping mapping;
  auto readerFunc = cast<func::FuncOp>(func->clone(mapping));
  readerFunc.setName((func.getName() + "__reader").str());

  // Collect compute ops and stores to erase.
  SmallVector<Operation *, 8> opsToErase;
  readerFunc.walk([&](Operation *op) {
    if (isComputeOp(op)) {
      opsToErase.push_back(op);
    } else if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
      if (isStoreOp(copyOp))
        opsToErase.push_back(op);
    }
  });

  // Erase compute ops in reverse order to handle dependencies
  for (Operation *op : llvm::reverse(opsToErase)) {
    // For linalg.matmul, the output is an "in-out" operand (read-modify-write).
    // Before erasing, we need to forward the output operand to users of the result
    // if any (though linalg.matmul with memref semantics doesn't produce results).
    if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
      // linalg.matmul with memref semantics has no results, safe to erase
      op->erase();
    } else {
      op->erase();
    }
  }

  return readerFunc;
}

/**
 * @brief Specialize a function for writer-only data movement.
 *
 * @details Clones the function with `__writer` suffix and erases all
 *          compute operations and load operations (memref.copy where the
 *          source is reinterpret_cast). Stores are retained to model
 *          CB/L1 -> DRAM traffic.
 *
 * @param func The original function to specialize.
 * @return The specialized writer function.
 */
static func::FuncOp makeWriterFunc(func::FuncOp func) {
  IRMapping mapping;
  auto writerFunc = cast<func::FuncOp>(func->clone(mapping));
  writerFunc.setName((func.getName() + "__writer").str());

  SmallVector<Operation *, 8> opsToErase;
  writerFunc.walk([&](Operation *op) {
    if (isComputeOp(op)) {
      opsToErase.push_back(op);
    } else if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
      if (isLoadOp(copyOp))
        opsToErase.push_back(op);
    }
  });

  for (Operation *op : llvm::reverse(opsToErase)) {
    if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
      (void)matmulOp;
      op->erase();
    } else {
      op->erase();
    }
  }

  return writerFunc;
}

/**
 * @brief Specializer class that manages function cloning and specialization.
 *
 * @details Follows the CoreSpecialize pattern from triton-tenstorrent.
 *          For each function, creates compute, reader, and writer
 *          specialized versions, inserts them into the module, and
 *          erases the original.
 */
class FunctionSpecializer {
public:
  FunctionSpecializer(ModuleOp module, func::FuncOp func)
      : module(module), originalFunc(func) {
    // Create specialized versions
    auto computeFunc = makeComputeFunc(func);
    auto readerFunc = makeReaderFunc(func);
    auto writerFunc = makeWriterFunc(func);

    // Attach TTKernel thread type attributes so downstream TTKernelToCpp
    // translation can recognize these as kernel entry points.
    auto *ctx = module.getContext();
    auto computeAttr = mlir::tt::ttkernel::ThreadTypeAttr::get(
        ctx, mlir::tt::ttkernel::ThreadType::Compute);
    auto nocAttr = mlir::tt::ttkernel::ThreadTypeAttr::get(
        ctx, mlir::tt::ttkernel::ThreadType::Noc);

    computeFunc->setAttr(mlir::tt::ttkernel::ThreadTypeAttr::name, computeAttr);
    readerFunc->setAttr(mlir::tt::ttkernel::ThreadTypeAttr::name, nocAttr);
    writerFunc->setAttr(mlir::tt::ttkernel::ThreadTypeAttr::name, nocAttr);

    // Insert specialized functions into the module (before the original)
    module.insert(func, computeFunc);
    module.insert(func, readerFunc);
    module.insert(func, writerFunc);
  }

private:
  ModuleOp module;
  func::FuncOp originalFunc;
};

} // namespace

void mlir::loom::specializeFunctionsForTTKernel(ModuleOp module) {
  // Collect functions to specialize (use make_early_inc_range since we modify
  // the module during iteration)
  for (func::FuncOp func :
       llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
    // Skip functions that are already specialized
    StringRef name = func.getName();
    if (name.ends_with("__compute") || name.ends_with("__reader") ||
        name.ends_with("__writer"))
      continue;

    // Skip external/declaration-only functions
    if (func.isExternal())
      continue;

    // Create specialized versions
    FunctionSpecializer specializer(module, func);

    // Erase the original function
    func.erase();
  }
}

LogicalResult mlir::loom::removeAllFunctionArguments(func::FuncOp func) {
  Block &entry = func.front();

  // Ensure no argument still has uses before erasing.
  for (BlockArgument arg : entry.getArguments()) {
    if (!arg.use_empty())
      return func.emitError()
             << "cannot erase arguments; argument " << arg.getArgNumber()
             << " still has uses";
  }

  // Erase arguments from last to first to avoid index shifting issues.
  for (int64_t idx = static_cast<int64_t>(entry.getNumArguments()) - 1;
       idx >= 0; --idx) {
    entry.eraseArgument(static_cast<unsigned>(idx));
  }

  auto funcType = func.getFunctionType();
  func.setType(FunctionType::get(func.getContext(), TypeRange(),
                                 funcType.getResults()));
  return success();
}