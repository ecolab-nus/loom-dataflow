// Standalone driver that annotates `memref.reinterpret_cast` ops with reuse
// information relative to surrounding affine/scf iterators.
//
// Usage:
//   tmd_triton_shared_annotate_reuse <input.mlir>

#include "reinterpret_cast_reuse.h"

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

#include "DataflowDialect.h.inc"
#include "DataflowOps.h.inc"

using namespace mlir;

int main(int argc, char **argv) {
  MLIRContext context;
  context.loadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::affine::AffineDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::tensor::TensorDialect>();
  context.loadDialect<mlir::linalg::LinalgDialect>();
  context.loadDialect<mlir::scf::SCFDialect>();
  context.loadDialect<mlir::bufferization::BufferizationDialect>();
  context.loadDialect<tmd::df::DataflowDialect>();

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
  pm.addPass(tmd::passes::createAnnotateReinterpretCastReusePass());
  if (failed(pm.run(*module))) {
    llvm::WithColor::error(llvm::errs()) << "Reuse annotation pass failed\n";
    return 2;
  }

  AsmState state(*module);
  module->print(llvm::outs(), state);
  llvm::outs() << "\n";
  return 0;
}
