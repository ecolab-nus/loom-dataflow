/**
 * @file utils.cpp
 * @brief Implementation of common utilities for function cloning.
 */

#include "utils.h"
#include "constraint_space_utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#include "hardware_info.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

// Include the generated Dataflow dialect headers
#include "DataflowDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.h.inc"
#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

// Include the generated Loom dialect headers
#include "LoomDialect.h.inc"
#include "mlir/Interfaces/ViewLikeInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace loom {
namespace utils {

ModuleOp getParentModule(func::FuncOp func) {
  Operation *parent = func->getParentOp();
  if (auto module = dyn_cast_or_null<ModuleOp>(parent)) {
    return module;
  }
  return nullptr;
}

namespace {

/// Private helper to clone a function and optionally apply a modifier.
/// Handles insertion point management and failure cleanup.
func::FuncOp cloneFunctionImpl(
    OpBuilder &builder, func::FuncOp originalFunc, llvm::StringRef newName,
    Operation *insertAfter, DictionaryAttr moduleAttrs,
    std::function<LogicalResult(func::FuncOp)> modifier = nullptr) {

  // Set insertion point
  if (insertAfter) {
    builder.setInsertionPointAfter(insertAfter);
  }

  // Create wrapper module if needed
  ModuleOp wrapperModule = nullptr;
  OpBuilder effectiveBuilder = builder;
  if (moduleAttrs) {
    wrapperModule = ModuleOp::create(originalFunc.getLoc());
    wrapperModule->setAttrs(moduleAttrs);
    builder.insert(wrapperModule);
    effectiveBuilder = OpBuilder(wrapperModule.getBodyRegion());
  }

  // Clone the function
  IRMapping mapping;
  auto clonedFunc =
      cast<func::FuncOp>(effectiveBuilder.clone(*originalFunc, mapping));
  clonedFunc.setName(newName);

  // Apply modifier if provided
  if (modifier) {
    if (failed(modifier(clonedFunc))) {
      if (wrapperModule) {
        wrapperModule.erase();
      } else {
        clonedFunc.erase();
      }
      return nullptr;
    }
  }

  // Update builder's insertion point to after the inserted operation
  builder.setInsertionPointAfter(wrapperModule ? (Operation *)wrapperModule
                                               : (Operation *)clonedFunc);

  return clonedFunc;
}

} // namespace

func::FuncOp cloneAndInsertFunction(OpBuilder &builder,
                                    func::FuncOp originalFunc,
                                    llvm::StringRef newName,
                                    Operation *insertAfter) {
  return cloneFunctionImpl(builder, originalFunc, newName, insertAfter,
                           nullptr);
}

func::FuncOp cloneAndInsertFunctionWithModuleWrapper(OpBuilder &builder,
                                                     func::FuncOp originalFunc,
                                                     llvm::StringRef newName,
                                                     DictionaryAttr moduleAttrs,
                                                     Operation *insertAfter) {
  return cloneFunctionImpl(builder, originalFunc, newName, insertAfter,
                           moduleAttrs);
}

func::FuncOp cloneModifyAndInsertFunction(
    OpBuilder &builder, func::FuncOp originalFunc, llvm::StringRef newName,
    std::function<LogicalResult(func::FuncOp)> modifier,
    Operation *insertAfter) {
  return cloneFunctionImpl(builder, originalFunc, newName, insertAfter, nullptr,
                           modifier);
}

func::FuncOp cloneModifyAndInsertFunctionWithModuleWrapper(
    OpBuilder &builder, func::FuncOp originalFunc, llvm::StringRef newName,
    DictionaryAttr moduleAttrs,
    std::function<LogicalResult(func::FuncOp)> modifier,
    Operation *insertAfter) {
  return cloneFunctionImpl(builder, originalFunc, newName, insertAfter,
                           moduleAttrs, modifier);
}

llvm::SmallVector<func::FuncOp> collectFunctions(ModuleOp module) {
  llvm::SmallVector<func::FuncOp> funcs;

  // Recursively collect functions from nested modules
  module.walk([&](func::FuncOp func) { funcs.push_back(func); });

  return funcs;
}

func::FuncOp cloneFuncWithConstraints(
    OpBuilder &builder, func::FuncOp originalFunc, llvm::StringRef newName,
    DictionaryAttr moduleAttrs, llvm::StringRef passName,
    std::function<LogicalResult(func::FuncOp, loom::ConstraintSpaceOp)>
        modifier,
    Operation *insertAfter) {

  // Set insertion point for the wrapper module
  if (insertAfter) {
    builder.setInsertionPointAfter(insertAfter);
  }

  // Create the wrapper module
  auto wrapperModule = ModuleOp::create(originalFunc.getLoc());
  builder.insert(wrapperModule);

  // Set the attributes on the wrapper module
  if (moduleAttrs) {
    wrapperModule->setAttrs(moduleAttrs);
  }

  // Set pass name attribute for provenance tracking
  loom::lcs::setPassNameAttr(wrapperModule, passName);

  // Create a builder for the wrapper module's body
  OpBuilder moduleBuilder(wrapperModule.getBodyRegion());

  // Find and clone the constraint space from the original function's parent
  // module
  loom::ConstraintSpaceOp originalCsOp = nullptr;
  if (auto parentModule = originalFunc->getParentOfType<ModuleOp>()) {
    originalCsOp = loom::lcs::findConstraintSpace(parentModule);
  }

  loom::ConstraintSpaceOp clonedCsOp = nullptr;
  if (originalCsOp) {
    clonedCsOp = loom::lcs::cloneConstraintSpace(moduleBuilder, originalCsOp);
  }

  // Clone the function into the wrapper module
  IRMapping mapping;
  auto clonedFunc =
      cast<func::FuncOp>(moduleBuilder.clone(*originalFunc, mapping));

  // Set the new name
  clonedFunc.setName(newName);

  // Apply the modifier
  if (clonedCsOp) {
    if (failed(modifier(clonedFunc, clonedCsOp))) {
      // Modifier failed, erase the wrapper module and return nullptr
      wrapperModule.erase();
      return nullptr;
    }

    // Check feasibility after modification
    if (!loom::lcs::isFeasible(clonedCsOp)) {
      // Constraint space is infeasible, erase the wrapper module
      wrapperModule.erase();
      return nullptr;
    }
  } else {
    // No constraint space found, just apply modifier with nullptr
    // This allows the function to work even without constraints
    if (failed(modifier(clonedFunc, nullptr))) {
      wrapperModule.erase();
      return nullptr;
    }
  }

  // Modifier succeeded and constraint space is feasible
  // Update insertion point to after the wrapper module
  builder.setInsertionPointAfter(wrapperModule);

  return clonedFunc;
}

StringRef traceToSymbolicVar(Value val) {
  if (!val)
    return "";

  // Handle direct loom.get_symbolic_block_size
  if (auto getSym = val.getDefiningOp<loom::GetSymbolicBlockSizeOp>()) {
    return getSym.getSymbolRef().getLeafReference().getValue();
  }

  // Handle arith.muli/addi/etc. if needed, but for now we follow the user's
  // sketch where block sizes are directly used from
  // loom.get_symbolic_block_size.

  return "";
}

llvm::SmallVector<AllocInfo> collectL1AllocInfos(func::FuncOp func) {
  llvm::SmallVector<AllocInfo> allocInfos;

  func.walk([&](loom::AlloccOp alloc) {
    // 1. Only care about allocations on @L1
    if (alloc.getMemory().getLeafReference() != "L1")
      return;

    // 2. Find element type from users (loom.init_tensor or loom.copy_to_tensor)
    Type elementType;
    for (auto user : alloc.getResult().getUsers()) {
      if (auto initTensor = dyn_cast<loom::InitTensorOp>(user)) {
        elementType = cast<RankedTensorType>(initTensor.getResult().getType())
                          .getElementType();
        break;
      }
      if (auto copyToTensor = dyn_cast<loom::CopyToTensorOp>(user)) {
        elementType = cast<RankedTensorType>(copyToTensor.getResult().getType())
                          .getElementType();
        break;
      }
    }

    if (!elementType)
      return;

    // Get base element size (Bytes)
    int64_t baseElemSize = elementType.getIntOrFloatBitWidth() / 8;

    AllocInfo info;
    // 3. Consider buffer_count (multi-buffering takes more space)
    info.elemSize = baseElemSize * alloc.getBufferCount();

    // 4. Trace dynamic dimensions
    auto dynamicOperands = alloc.getDynamicSizes();
    llvm::SmallVector<StringRef> symbolicVars;
    llvm::SmallVector<std::pair<int64_t, StringRef>> ceildivs;

    for (Value val : dynamicOperands) {
      if (auto ceildiv = val.getDefiningOp<arith::CeilDivSIOp>()) {
        // Handle (Constant / SymbolicVar) pattern for hoisted block
        // cancellation
        int64_t numerator = -1;
        if (auto constOp =
                ceildiv.getLhs().getDefiningOp<arith::ConstantIndexOp>()) {
          numerator = constOp.value();
        } else if (auto constIntOp =
                       ceildiv.getLhs().getDefiningOp<arith::ConstantIntOp>()) {
          numerator = constIntOp.value();
        }

        if (numerator > 0) {
          StringRef denominator = traceToSymbolicVar(ceildiv.getRhs());
          if (!denominator.empty()) {
            ceildivs.push_back({numerator, denominator});
            continue; // Already handled as ceildiv pattern
          }
        }
      }

      // Normal symbolic variable tracing
      StringRef symName = traceToSymbolicVar(val);
      if (!symName.empty()) {
        symbolicVars.push_back(symName);
      }
    }

    // 5. Cancellation logic for hoisted blocks: block_K * (K_total / block_K)
    // -> K_total
    for (auto &pair : ceildivs) {
      int64_t numerator = pair.first;
      StringRef denominator = pair.second;
      for (auto it = symbolicVars.begin(); it != symbolicVars.end(); ++it) {
        if (*it == denominator) {
          symbolicVars.erase(it);
          info.elemSize *= numerator; // Merge total size multiplication
          break;
        }
      }
    }

    info.dims = symbolicVars;

    // 6. Record if it has symbolic dims or total size exceeds single element
    // (fixed-size multi-buffer or hoisted buffer)
    if (!info.dims.empty() || info.elemSize > baseElemSize) {
      allocInfos.push_back(std::move(info));
    }
  });

  return allocInfos;
}

} // namespace utils

void utils::composeAndCanonicalizeAffineApplies(func::FuncOp func) {
  SmallVector<affine::AffineApplyOp> applies;
  func.walk([&](affine::AffineApplyOp op) { applies.push_back(op); });
  for (affine::AffineApplyOp op : applies) {
    OpBuilder b(op);
    AffineMap map = op.getAffineMap();
    SmallVector<Value> operands(op.getOperands().begin(),
                                op.getOperands().end());
    affine::fullyComposeAffineMapAndOperands(&map, &operands);
    affine::canonicalizeMapAndOperands(&map, &operands);
    bool sameMap = (map == op.getAffineMap());
    bool sameOperands =
        operands.size() == op.getNumOperands() &&
        std::equal(operands.begin(), operands.end(), op.getOperands().begin());
    if (sameMap && sameOperands)
      continue;
    auto newOp = b.create<affine::AffineApplyOp>(op.getLoc(), map, operands);
    op.replaceAllUsesWith(newOp.getResult());
    op.erase();
  }

  SmallVector<Operation *> toErase;
  func.walk([&](Operation *op) {
    if (mlir::isOpTriviallyDead(op))
      toErase.push_back(op);
  });
  for (Operation *op : toErase)
    op->erase();
}

} // namespace loom
