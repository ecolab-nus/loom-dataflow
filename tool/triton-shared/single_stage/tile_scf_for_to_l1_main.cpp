// Standalone driver that tiles scf.for loops to fit within the single df.memory (L1).
//
// Usage:
//   tile_scf_for_to_l1 <input.mlir>

#include "tile_scf_for_to_l1.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  os << "  The input must be fully bufferized (no tensor types).\n";
  os << "  The module must declare exactly one df.memory.\n";
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
  context.loadDialect<mlir::linalg::LinalgDialect>();
  context.loadDialect<mlir::scf::SCFDialect>();
  context.loadDialect<loom::df::DataflowDialect>();

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
  pm.addPass(loom::passes::createTileScfForToL1Pass());
  if (failed(pm.run(*module))) {
    llvm::WithColor::error(llvm::errs()) << "Tile scf.for to L1 pass failed\n";
    return 2;
  }

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  // Reorder DF ops to the front (stable order) and print.
  {
    Block &body = *module->getBody();
    SmallVector<Operation *, 16> dfOps;
    for (Operation &op : body) {
      Dialect *dialect = op.getDialect();
      if (dialect && dialect->getNamespace() == StringRef("df"))
        dfOps.push_back(&op);
    }
    for (Operation *op : llvm::reverse(dfOps))
      op->moveBefore(&body.front());
  }
  module->print(llvm::outs(), flags);
  llvm::outs() << "\n";
  return 0;
}

