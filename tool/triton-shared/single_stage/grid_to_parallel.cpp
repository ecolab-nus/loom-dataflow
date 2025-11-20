// Standalone driver to run the Triton-shared grid-to-parallel pass.
//
// Usage:
//   tmd_triton_shared_grid_to_parallel <input.mlir> [--df <df.mlir>]

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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

#include "DataflowDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.h.inc"
#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

using namespace mlir;

static llvm::cl::opt<std::string>
    clInput("input", llvm::cl::Positional,
            llvm::cl::desc("Input MLIR file"),
            llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    clDfInput("df",
              llvm::cl::desc("Path to DF MLIR file (spatial description)"),
              llvm::cl::value_desc("filename"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TMD Triton-shared grid-to-parallel\n");

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
  context.loadDialect<tmd::df::DataflowDialect>();

  const char *filename = clInput.c_str();
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

  // Parse DF module if provided and merge it into the main module.
  if (!clDfInput.empty()) {
    llvm::SourceMgr dfSm;
    auto dfFile = mlir::openInputFile(clDfInput);
    if (!dfFile) {
      llvm::WithColor::error(llvm::errs())
          << "Failed to open DF input file: " << clDfInput << "\n";
      return 1;
    }
    dfSm.AddNewSourceBuffer(std::move(dfFile), llvm::SMLoc());
    OwningOpRef<ModuleOp> dfModule = parseSourceFile<ModuleOp>(dfSm, &context);
    if (!dfModule) {
      llvm::WithColor::error(llvm::errs()) << "Failed to parse DF MLIR module\n";
      return 1;
    }

    // Merge DF operations into the main module at the beginning.
    OpBuilder builder(module->getBodyRegion());
    builder.setInsertionPointToStart(module->getBody());
    IRMapping mapping;
    for (Operation &op : *dfModule->getBody())
      builder.clone(op, mapping);
  }

  PassManager pm(&context);
  pm.addPass(tmd::passes::createTritonSharedGridToParallelPass());

  if (failed(pm.run(*module))) {
    llvm::WithColor::error(llvm::errs()) << "Grid-to-parallel pass failed\n";
    return 2;
  }

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  module->print(llvm::outs(), flags);
  llvm::outs() << "\n";
  return 0;
}
