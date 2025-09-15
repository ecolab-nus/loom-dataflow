// Standalone driver to run the Triton-shared grid-to-parallel pass.
//
// Usage:
//   tmd_triton_shared_grid_to_parallel <input.mlir>

#include "triton_shared_grid_to_parallel.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;

int main(int argc, char **argv) {
  MLIRContext context;
  (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::affine::AffineDialect>();
  context.loadDialect<mlir::tensor::TensorDialect>();
  context.loadDialect<mlir::linalg::LinalgDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::scf::SCFDialect>();
  context.loadDialect<mlir::bufferization::BufferizationDialect>();

  const char *filename = argc > 1 ? argv[1] : "-";
  llvm::SourceMgr sm;
  auto file = mlir::openInputFile(filename);
  if (!file) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open input file: " << filename << "\n";
    return 1;
  }
  sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sm, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs()) << "Failed to parse MLIR module\n";
    return 1;
  }

  PassManager pm(&context);
  pm.addPass(tmd::passes::createTritonSharedGridToParallelPass());

  if (failed(pm.run(*module))) {
    llvm::WithColor::error(llvm::errs()) << "Grid-to-parallel pass failed\n";
    return 2;
  }

  AsmState state(*module);
  module->print(llvm::outs(), state);
  llvm::outs() << "\n";
  return 0;
}
