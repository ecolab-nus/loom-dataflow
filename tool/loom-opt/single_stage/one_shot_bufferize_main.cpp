// Standalone driver to run the LOOM One-Shot Bufferization.
//
// Usage:
//   one_shot_bufferize --input <input.mlir>

#include "Passes.h"
#include "Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "ADLDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"
#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;
using namespace loom;

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> inputFilename(
      "input", llvm::cl::desc("Input MLIR file"),
      llvm::cl::value_desc("filename"), llvm::cl::init("-"));

  llvm::cl::ParseCommandLineOptions(argc, argv, "Loom OSB Tool\n");

  MLIRContext context;
  // Register Standard BufferizableOpInterface external models
  DialectRegistry registry;
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerSubsetOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);

  context.appendDialectRegistry(registry);

  // Load necessary dialects
  context.loadDialect<LoomDialect, adl::ADLDialect, func::FuncDialect,
                      memref::MemRefDialect, scf::SCFDialect,
                      arith::ArithDialect, linalg::LinalgDialect,
                      tensor::TensorDialect, affine::AffineDialect,
                      bufferization::BufferizationDialect>();

  // Register Loom's BufferizableOpInterface external models directly on context
  loom::registerBufferizableOpInterfaceExternalModels(&context);

  // Parse input
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << inputFilename << ": "
                 << ec.message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Could not parse MLIR file\n";
    return 1;
  }

  PassManager pm(&context);
  pm.addPass(loom::passes::createLowerAffineWithAttrPass());

  // Configure OSB options
  bufferization::OneShotBufferizePassOptions options;
  options.allowUnknownOps = false;
  options.bufferizeFunctionBoundaries = true;
  options.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;

  // OSB Pass
  pm.addPass(bufferization::createOneShotBufferizePass(options));

  // Run OSB on nested modules too
  pm.nest<ModuleOp>().addPass(
      bufferization::createOneShotBufferizePass(options));

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (failed(pm.run(*module))) {
    llvm::WithColor::error(llvm::errs()) << "One-Shot Bufferization failed\n";
    return 2;
  }

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  module->print(llvm::outs(), flags);
  llvm::outs() << "\n";
  return 0;
}
