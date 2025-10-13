// Single-stage: annotate reuse and explore alloc/copy mapping choices.

#include "explore_alloc_copy_mapping.h"
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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

#include "DataflowDialect.h.inc"

using namespace mlir;

static llvm::cl::opt<std::string>
    clInput("input", llvm::cl::desc("Path to input MLIR (DF+funcs)"),
            llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<bool> clAnalysisOnly(
    "analysis-only",
    llvm::cl::desc("Only attach tmd.copy.candidates; do not clone"),
    llvm::cl::init(false));

static llvm::cl::opt<long long>
    clMaxVariants("max-variants",
                  llvm::cl::desc("Max clones per function (-1 = unlimited)"),
                  llvm::cl::init(-1));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Single-stage alloc/copy mapping explorer\n");

  // Setup context and register required dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                  mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::bufferization::BufferizationDialect,
                  tmd::df::DataflowDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Parse input module that includes DF hardware decls and functions.
  llvm::SourceMgr sm;
  auto file = mlir::openInputFile(clInput);
  if (!file) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open input file: " << clInput << "\n";
    return 1;
  }
  sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sm, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs()) << "Failed to parse MLIR module\n";
    return 1;
  }

  // 1) Annotate reuse on reinterpret_cast.
  PassManager pm(&context);
  pm.addPass(tmd::passes::createAnnotateReinterpretCastReusePass());
  if (failed(pm.run(*module))) {
    llvm::WithColor::error(llvm::errs()) << "Reuse annotation failed\n";
    return 2;
  }

  // 2) Explore alloc/copy mappings.
  PassManager pm2(&context);
  pm2.addPass(tmd::passes::createExploreAllocCopyMappingPass(clAnalysisOnly,
                                                             clMaxVariants));
  if (failed(pm2.run(*module))) {
    llvm::WithColor::error(llvm::errs()) << "Mapping exploration failed\n";
    return 3;
  }

  // Print result.
  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  module->print(llvm::outs(), flags);
  llvm::outs() << "\n";
  return 0;
}
