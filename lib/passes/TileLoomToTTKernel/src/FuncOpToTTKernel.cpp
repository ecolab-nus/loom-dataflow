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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

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

