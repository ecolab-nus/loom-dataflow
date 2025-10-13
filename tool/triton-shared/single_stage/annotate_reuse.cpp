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
#include "llvm/Support/raw_ostream.h"

#include "DataflowDialect.h.inc"
#include "DataflowOps.h.inc"

using namespace mlir;

/// Print usage information for this tool.
///
/// \param os        The output stream to write usage text to.
/// \param progName  The program name as invoked (argv[0]).
static void printUsage(llvm::raw_ostream &os, const char *progName) {
  os << "Usage:\n";
  os << "  " << progName << " [<input.mlir>]\n\n";
  os << "Options:\n";
  os << "  -h, --help    Show this help message and exit\n\n";
  os << "Notes:\n";
  os << "  If no input is provided, or '-' is used, reads from stdin.\n";
}

int main(int argc, char **argv) {
  // Handle help and basic argument validation before doing any work.
  if (argc > 2) {
    llvm::WithColor::error(llvm::errs()) << "Unexpected arguments.\n";
    printUsage(llvm::errs(), argv[0]);
    return 1;
  }
  if (argc == 2) {
    llvm::StringRef arg1(argv[1]);
    if (arg1 == "-h" || arg1 == "--help" || arg1 == "-help" || arg1 == "-?") {
      printUsage(llvm::outs(), argv[0]);
      return 0;
    }
  }

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
