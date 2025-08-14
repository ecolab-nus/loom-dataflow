#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include <iostream>

using namespace mlir;

int main(int argc, char **argv) {
  MLIRContext context;
  (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::func::FuncDialect, mlir::affine::AffineDialect,
                      mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();

  // Load file
  llvm::SourceMgr sourceMgr;
  const char *filename = argc > 1 ? argv[1] : "-";
  auto file = mlir::openInputFile(filename);
  if (!file) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open input file: " << filename << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  // Parse
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs()) << "Failed to parse MLIR module\n";
    return 1;
  }

  // Count affine ops
  size_t numFor = 0, numIf = 0, numLoad = 0, numStore = 0, numApply = 0;
  module->walk([&](Operation *op) {
    if (llvm::isa<mlir::affine::AffineForOp>(op))
      ++numFor;
    else if (llvm::isa<mlir::affine::AffineIfOp>(op))
      ++numIf;
    else if (llvm::isa<mlir::affine::AffineLoadOp>(op))
      ++numLoad;
    else if (llvm::isa<mlir::affine::AffineStoreOp>(op))
      ++numStore;
    else if (llvm::isa<mlir::affine::AffineApplyOp>(op))
      ++numApply;
  });

  std::cout << "affine.for:   " << numFor << "\n";
  std::cout << "affine.if:    " << numIf << "\n";
  std::cout << "affine.load:  " << numLoad << "\n";
  std::cout << "affine.store: " << numStore << "\n";
  std::cout << "affine.apply: " << numApply << "\n";

  return 0;
}
