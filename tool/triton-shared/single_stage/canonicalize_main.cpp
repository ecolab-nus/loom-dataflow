// Standalone driver that materializes and canonicalizes IR operations.
//
// Usage:
//   loom_triton_shared_canonicalize --input <input.mlir>
//   loom_triton_shared_canonicalize --input -  (reads from stdin)

#include "loom_to_memref.h"
#include "materialize.h"
#include "staticize_types.h"

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
#include "DataflowOps.h.inc"
#include "LoomDialect.h.inc"
#include "triton_shared_affinize.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    clInput("input", llvm::cl::desc("Path to input MLIR file (use '-' for stdin)"),
            llvm::cl::value_desc("filename"), llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "LOOM Triton-shared canonicalize pass\n");

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
  context.loadDialect<loom::df::DataflowDialect>();
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
  /// Step 1: Materialize - Replace loom.get_module_attribute with arith.constant.
  pm.addPass(loom::passes::createMaterializePass());
  /// Step 2: Staticize - Convert dynamic memref/tensor types to static types.
  pm.addPass(loom::passes::createStaticizeTypesPass());
  /// Step 3: Lower - Lower loom operations to memref dialect.
  pm.addPass(loom::passes::createLoomToMemRefLoweringPass());
  /// Step 4: Affinize - Convert index arithmetic to affine IR.
  pm.addPass(loom::passes::createTritonSharedAffinizePass());
  if (failed(pm.run(*module))) {
    llvm::WithColor::error(llvm::errs()) << "Canonicalize pass failed\n";
    return 2;
  }

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  module->print(llvm::outs(), flags);
  llvm::outs() << "\n";
  return 0;
}

