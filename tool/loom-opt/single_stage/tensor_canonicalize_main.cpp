// Standalone driver to run the LOOM tensor-canonicalize passes.
//
// Usage:
//   tensor_canonicalize --input <input.mlir>

#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
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

#include "LoomDialect.h.inc"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    clInput("input", llvm::cl::desc("Path to input MLIR file"),
            llvm::cl::value_desc("filename"), llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "LOOM tensor-canonicalize pass driver\n");

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
  context.loadDialect<mlir::cf::ControlFlowDialect>();
  context.loadDialect<loom::LoomDialect>();

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

  PassManager pm(&context);

  // Preprocessing: fuse elementwise generics to make patterns visible and clean
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
  pm.addPass(mlir::createLinalgFoldUnitExtentDimsPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Our destination specialization - now works on simplified/fused IR
  pm.addPass(loom::passes::createLinalgDestinationSpecializationPass());

  // Postprocessing: remove dead producers and final cleanup
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Eliminate redundant tensor.extract_slice surviving fusion/canonicalization
  pm.addPass(loom::passes::createFoldRedundantExtractSlicePass());
  pm.addPass(mlir::createCanonicalizerPass());

  // De-CSE: clone and sink all linalg.fill ops to ensure unique SSA chains
  // for initialized tensors, eliminating cross-scope fill sharing.
  pm.addPass(loom::passes::createSinkFillOpsPass());

  if (failed(pm.run(*module))) {
    llvm::WithColor::error(llvm::errs())
        << "LOOM tensor canonicalization pipeline failed\n";
    return 2;
  }

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  module->print(llvm::outs(), flags);
  llvm::outs() << "\n";
  return 0;
}
