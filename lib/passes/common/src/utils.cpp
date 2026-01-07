/**
 * @file utils.cpp
 * @brief Implementation of common utilities for function cloning.
 */

#include "utils.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace loom {
namespace utils {

func::FuncOp cloneAndInsertFunction(
    OpBuilder &builder,
    func::FuncOp originalFunc,
    llvm::StringRef newName,
    Operation *insertAfter) {
  
  // Set insertion point
  if (insertAfter) {
    builder.setInsertionPointAfter(insertAfter);
  }
  // If insertAfter is nullptr, use the current insertion point
  
  // Clone the function
  IRMapping mapping;
  auto clonedFunc = cast<func::FuncOp>(builder.clone(*originalFunc, mapping));
  
  // Set the new name
  clonedFunc.setName(newName);
  
  // Update insertion point to after the cloned function
  builder.setInsertionPointAfter(clonedFunc);
  
  return clonedFunc;
}

func::FuncOp cloneModifyAndInsertFunction(
    OpBuilder &builder,
    func::FuncOp originalFunc,
    llvm::StringRef newName,
    std::function<LogicalResult(func::FuncOp)> modifier,
    Operation *insertAfter) {
  
  // Clone the function (don't update insertion point yet)
  if (insertAfter) {
    builder.setInsertionPointAfter(insertAfter);
  }
  
  IRMapping mapping;
  auto clonedFunc = cast<func::FuncOp>(builder.clone(*originalFunc, mapping));
  
  // Set the new name
  clonedFunc.setName(newName);
  
  // Apply the modifier
  if (failed(modifier(clonedFunc))) {
    // Modifier failed, erase the cloned function and return nullptr
    clonedFunc.erase();
    return nullptr;
  }
  
  // Modifier succeeded, update insertion point to after the cloned function
  builder.setInsertionPointAfter(clonedFunc);
  
  return clonedFunc;
}

llvm::SmallVector<func::FuncOp> collectFunctions(ModuleOp module) {
  llvm::SmallVector<func::FuncOp> funcs;
  
  for (auto func : module.getOps<func::FuncOp>()) {
    funcs.push_back(func);
  }
  
  return funcs;
}

} // namespace utils
} // namespace loom

