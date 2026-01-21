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

func::FuncOp loom::CompileArgTracker::getParentFunc(Operation *op) const {
  return op->getParentOfType<func::FuncOp>();
}

loom::FunctionTrackingData *
loom::CompileArgTracker::getOrCreateFuncData(Operation *op) {
  func::FuncOp func = getParentFunc(op);
  assert(func && "Operation must be inside a function");
  return getOrCreateFuncData(func);
}

loom::FunctionTrackingData *
loom::CompileArgTracker::getOrCreateFuncData(func::FuncOp func) {
  auto it = funcToData.find(func);
  if (it != funcToData.end())
    return &it->second;
  return &funcToData.try_emplace(func, FunctionTrackingData{}).first->second;
}

loom::FunctionTrackingData *
loom::CompileArgTracker::getFuncData(func::FuncOp func) {
  auto it = funcToData.find(func);
  if (it != funcToData.end())
    return &it->second;
  return nullptr;
}

int64_t loom::CompileArgTracker::getOrCreateAlloc(memref::AllocOp alloc) {
  auto *funcData = getOrCreateFuncData(alloc);
  Value v = alloc.getResult();
  auto it = funcData->allocToCbIndex.find(v);
  if (it != funcData->allocToCbIndex.end())
    return it->second;
  // Assign next available index for allocs not associated with a function argument.
  int64_t cbIndex = funcData->nextCompileArgIndex++;
  funcData->allocToCbIndex.try_emplace(v, cbIndex);
  return cbIndex;
}

loom::CompileArgTracker::MemrefData
loom::CompileArgTracker::getOrCreate(Value inputMemref, memref::AllocOp alloc) {
  auto *funcData = getOrCreateFuncData(alloc);
  Value allocResult = alloc.getResult();
  
  // Check if the input memref was pre-recorded (e.g., as a function argument).
  // If so, use the pre-assigned (cbIndex, baseAddrIndex) pair.
  auto it = funcData->inputToData.find(inputMemref);
  if (it != funcData->inputToData.end()) {
    int64_t cbIndex = it->second.first;
    int64_t baseAddrIndex = it->second.second;
    
    // Record the alloc's CB index for later use.
    funcData->allocToCbIndex.try_emplace(allocResult, cbIndex);
    
    return MemrefData{cbIndex, baseAddrIndex};
  }
  
  // Not pre-recorded - assign new indices.
  int64_t cbIndex = getOrCreateAlloc(alloc);
  int64_t baseAddrIndex = cbIndex; // For non-function-arg memrefs, use same index
  funcData->inputToData.try_emplace(inputMemref,
                                    std::make_pair(cbIndex, baseAddrIndex));
  return MemrefData{cbIndex, baseAddrIndex};
}

loom::CompileArgTracker::MemrefData
loom::CompileArgTracker::getOrCreateOutput(Value outputMemref, Operation *op) {
  auto *funcData = getOrCreateFuncData(op);
  
  // Check if the output memref was pre-recorded as an input (function argument).
  // If so, use the pre-assigned (cbIndex, baseAddrIndex) pair.
  auto inputIt = funcData->inputToData.find(outputMemref);
  if (inputIt != funcData->inputToData.end()) {
    int64_t cbIndex = inputIt->second.first;
    int64_t baseAddrIndex = inputIt->second.second;
    funcData->outputToData.try_emplace(outputMemref,
                                       std::make_pair(cbIndex, baseAddrIndex));
    return MemrefData{cbIndex, baseAddrIndex};
  }
  
  // Check if already tracked as output
  auto it = funcData->outputToData.find(outputMemref);
  if (it != funcData->outputToData.end())
    return MemrefData{it->second.first, it->second.second};
  
  // Not pre-recorded - assign next available index
  int64_t cbIndex = funcData->nextCompileArgIndex++;
  funcData->outputToData.try_emplace(outputMemref,
                                     std::make_pair(cbIndex, cbIndex));
  return MemrefData{cbIndex, cbIndex};
}

int64_t loom::CompileArgTracker::getOrCreateIndex(Value indexValue,
                                                  Operation *op) {
  auto *funcData = getOrCreateFuncData(op);
  auto it = funcData->indexToArgIndex.find(indexValue);
  if (it != funcData->indexToArgIndex.end())
    return it->second;
  int64_t argIndex = funcData->nextCompileArgIndex++;
  funcData->indexToArgIndex.try_emplace(indexValue, argIndex);
  return argIndex;
}

//===----------------------------------------------------------------------===//
// replaceFuncArgsWithCompileArgs Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::loom::replaceFuncArgsWithCompileArgs(
    func::FuncOp func, std::shared_ptr<CompileArgTracker> tracker,
    TypeConverter &typeConverter, OpBuilder &rewriter) {
  Block &entry = func.front();
  Location loc = func.getLoc();

  // Save insertion point and set to start of function body.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&entry);

  // Get or create tracking data for this function.
  auto *funcData = tracker->getOrCreateFuncData(func);

  // First pass: calculate the compile-arg index for each function argument.
  // Memref args use TWO indices (CB + base addr), index args use ONE index.
  // This preserves the argument order while accounting for the dual indices.
  int64_t currentIndex = 0;
  for (BlockArgument arg : entry.getArguments()) {
    Type argType = arg.getType();

    if (isa<MemRefType, UnrankedMemRefType>(argType)) {
      // Memref type: assign two consecutive indices (CB, base addr).
      int64_t cbIndex = currentIndex;
      int64_t baseAddrIndex = currentIndex + 1;
      funcData->inputToData.try_emplace(arg, std::make_pair(cbIndex, baseAddrIndex));
      currentIndex += 2;
    } else if (argType.isIndex()) {
      // Index type: assign one index.
      funcData->indexToArgIndex.try_emplace(arg, currentIndex);
      currentIndex += 1;
    } else {
      // Other types: assign one index.
      currentIndex += 1;
    }
  }

  // Set nextCompileArgIndex to continue after all function arguments.
  funcData->nextCompileArgIndex = currentIndex;

  // Second pass: emit GetCompileArgValOp for index arguments.
  for (BlockArgument arg : entry.getArguments()) {
    Type argType = arg.getType();

    if (argType.isIndex()) {
      // Get the pre-assigned index.
      auto it = funcData->indexToArgIndex.find(arg);
      assert(it != funcData->indexToArgIndex.end());
      int64_t argIndex = it->second;
      
      auto idxAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(argIndex));
      auto compileArg =
          rewriter.create<GetCompileArgValOp>(loc, rewriter.getI32Type(), idxAttr);
      // Replace all uses of the index argument with the compile-arg value.
      // Need to cast i32 back to index for compatibility.
      auto indexCast = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), compileArg);
      arg.replaceAllUsesWith(indexCast);
    }
    // Memref types are handled by the memory conversion patterns.
  }

  return success();
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