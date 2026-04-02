// Single-stage: enumerate interconnect broadcast choices for copy operations.

#include "Passes.h"

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

#include "ADLDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"
#include "LoomDialect.h.inc"

using namespace mlir;

static llvm::cl::opt<std::string>
    clInput("input", llvm::cl::desc("Path to input MLIR (DF+funcs)"),
            llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<bool> clAnalysisOnly(
    "analysis-only",
    llvm::cl::desc("Only attach loom.copy.candidates; do not clone"),
    llvm::cl::init(false));

// No max-variants: we always enumerate all combinations.

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Single-stage copy broadcast enumerator\n");

  // Setup context and register required dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                  mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::bufferization::BufferizationDialect,
                  adl::ADLDialect, loom::LoomDialect>();
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

  PassManager pm1(&context);
  pm1.addPass(loom::passes::createEnumerateCopyBroadcastPass(clAnalysisOnly));
  if (failed(pm1.run(*module))) {
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
