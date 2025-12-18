// Standalone driver to run the Triton-shared affinization pass.
//
// Usage:
//   loom_triton_shared_affinize --ttshared <input.mlir>

#include "triton_shared_affinize.h"

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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

// Dataflow dialect for parsing df module sections (if present in file).
#include "DataflowDialect.h.inc"
#include "DataflowOps.h.inc"
// Loom dialect for parsing loom operations
#include "LoomDialect.h.inc"
#include "LoomOps.h.inc"

using namespace mlir;

static llvm::cl::opt<std::string>
    clTTSharedInput("ttshared", llvm::cl::desc("Path to ttshared MLIR file"),
                    llvm::cl::value_desc("filename"), llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "LOOM Triton-shared affinization\n");

  MLIRContext context;
  (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::tensor::TensorDialect>();
  context.loadDialect<mlir::linalg::LinalgDialect>();
  context.loadDialect<mlir::scf::SCFDialect>();
  context.loadDialect<mlir::bufferization::BufferizationDialect>();
  context.loadDialect<mlir::affine::AffineDialect>();
  context.loadDialect<loom::df::DataflowDialect>();
  context.loadDialect<loom::LoomDialect>();

  llvm::SourceMgr sm;
  auto file = mlir::openInputFile(clTTSharedInput);
  if (!file) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open input file: " << clTTSharedInput << "\n";
    return 1;
  }
  sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sm, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs()) << "Failed to parse MLIR module\n";
    return 1;
  }

  PassManager pm(&context);
  pm.addPass(loom::passes::createTritonSharedAffinizePass());

  if (failed(pm.run(*module))) {
    llvm::WithColor::error(llvm::errs()) << "Affinization pass failed\n";
    return 2;
  }

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  module->print(llvm::outs(), flags);
  llvm::outs() << "\n";
  return 0;
}
