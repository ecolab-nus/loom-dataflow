/**
 * @file utils.cpp
 * @brief Implementation of common utilities for function cloning.
 */

#include "utils.h"
#include "constraint_space_utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

// Include the generated Loom dialect headers for ConstraintSpaceOp
#include "LoomDialect.h.inc"
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

  func.walk([&](loom::AllocOp alloc) {
    // Only care about allocations on @L1
    if (alloc.getMemory().getLeafReference() != "L1")
      return;

    auto memrefType = cast<MemRefType>(alloc.getResult().getType());
    AllocInfo info;
    info.elemSize = memrefType.getElementTypeBitWidth() / 8;

    // Trace dynamic dimensions
    auto dynamicOperands = alloc.getDynamicSizes();
    llvm::SmallVector<StringRef> symbolicVars;
    llvm::SmallVector<std::pair<int64_t, StringRef>> ceildivs;

    for (Value val : dynamicOperands) {
      if (auto getSym = val.getDefiningOp<loom::GetSymbolicBlockSizeOp>()) {
        symbolicVars.push_back(
            getSym.getSymbolRef().getLeafReference().getValue());
      } else if (auto ceildiv = val.getDefiningOp<arith::CeilDivSIOp>()) {
        // Handle (Constant / SymbolicVar) pattern
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
          }
        }
      } else {
        StringRef symName = traceToSymbolicVar(val);
        if (!symName.empty()) {
          symbolicVars.push_back(symName);
        }
      }
    }

    // Cancellation logic for hoisted blocks: block_K * (K_total / block_K) ->
    // K_total
    for (auto &pair : ceildivs) {
      int64_t numerator = pair.first;
      StringRef denominator = pair.second;
      bool cancelled = false;
      for (auto it = symbolicVars.begin(); it != symbolicVars.end(); ++it) {
        if (*it == denominator) {
          symbolicVars.erase(it);
          info.elemSize *= numerator;
          cancelled = true;
          break;
        }
      }
      if (!cancelled) {
        // If not cancelled, it remains as a symbolic part (though complex)
        // For now, we prioritize the cancellation case per requirements.
      }
    }

    info.dims = symbolicVars;
    if (!info.dims.empty() ||
        info.elemSize > (int64_t)(memrefType.getElementTypeBitWidth() / 8)) {
      allocInfos.push_back(std::move(info));
    }
  });

  return allocInfos;
}

} // namespace utils
} // namespace loom
