/**
 * @file utils.cpp
 * @brief Implementation of common utilities for function cloning.
 */

#include "utils.h"
#include "mlir/IR/IRMapping.h"

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

func::FuncOp cloneAndInsertFunctionWithModuleWrapper(
    OpBuilder &builder,
    func::FuncOp originalFunc,
    llvm::StringRef newName,
    DictionaryAttr moduleAttrs,
    Operation *insertAfter) {
  
  // Set insertion point for the wrapper module
  if (insertAfter) {
    builder.setInsertionPointAfter(insertAfter);
  }
  // If insertAfter is nullptr, use the current insertion point
  
  // Create the wrapper module
  auto wrapperModule = builder.create<ModuleOp>(originalFunc.getLoc());
  
  // Set the attributes on the wrapper module
  if (moduleAttrs) {
    wrapperModule->setAttrs(moduleAttrs);
  }
  
  // Create a builder for the wrapper module's body
  OpBuilder moduleBuilder(wrapperModule.getBodyRegion());
  
  // Clone the function into the wrapper module
  IRMapping mapping;
  auto clonedFunc = cast<func::FuncOp>(moduleBuilder.clone(*originalFunc, mapping));
  
  // Set the new name
  clonedFunc.setName(newName);
  
  // Update insertion point to after the wrapper module
  builder.setInsertionPointAfter(wrapperModule);
  
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

func::FuncOp cloneModifyAndInsertFunctionWithModuleWrapper(
    OpBuilder &builder,
    func::FuncOp originalFunc,
    llvm::StringRef newName,
    DictionaryAttr moduleAttrs,
    std::function<LogicalResult(func::FuncOp)> modifier,
    Operation *insertAfter) {
  
  // Set insertion point for the wrapper module
  if (insertAfter) {
    builder.setInsertionPointAfter(insertAfter);
  }
  
  // Create the wrapper module
  auto wrapperModule = builder.create<ModuleOp>(originalFunc.getLoc());
  
  // Set the attributes on the wrapper module
  if (moduleAttrs) {
    wrapperModule->setAttrs(moduleAttrs);
  }
  
  // Create a builder for the wrapper module's body
  OpBuilder moduleBuilder(wrapperModule.getBodyRegion());
  
  // Clone the function into the wrapper module
  IRMapping mapping;
  auto clonedFunc = cast<func::FuncOp>(moduleBuilder.clone(*originalFunc, mapping));
  
  // Set the new name
  clonedFunc.setName(newName);
  
  // Apply the modifier
  if (failed(modifier(clonedFunc))) {
    // Modifier failed, erase both the function and wrapper module
    wrapperModule.erase();
    return nullptr;
  }
  
  // Modifier succeeded, update insertion point to after the wrapper module
  builder.setInsertionPointAfter(wrapperModule);
  
  return clonedFunc;
}

llvm::SmallVector<func::FuncOp> collectFunctions(ModuleOp module) {
  llvm::SmallVector<func::FuncOp> funcs;
  
  // Recursively collect functions from nested modules
  module.walk([&](func::FuncOp func) {
    funcs.push_back(func);
  });
  
  return funcs;
}

} // namespace utils
} // namespace loom

