// Standalone driver to parse and dump compute MLIR files from a directory.
//
// Usage:
//   parse_compute_ops --input-dir <directory>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "LoomDialect.h.inc"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    clInputDir("input-dir",
               llvm::cl::desc("Path to directory containing .mlir files"),
               llvm::cl::value_desc("directory"), llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "LOOM compute MLIR parse/dump tool\n");

  MLIRContext context;
  (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::tensor::TensorDialect>();
  context.loadDialect<mlir::linalg::LinalgDialect>();
  context.loadDialect<mlir::math::MathDialect>();
  context.loadDialect<loom::LoomDialect>();

  std::error_code ec;
  int failures = 0;
  int parsed = 0;

  for (llvm::sys::fs::directory_iterator dir(clInputDir, ec), end;
       dir != end && !ec; dir.increment(ec)) {
    llvm::StringRef path = dir->path();
    if (!path.ends_with(".mlir"))
      continue;

    llvm::outs() << "=== Parsing: " << path << " ===\n";

    llvm::SourceMgr sm;
    auto file = mlir::openInputFile(path);
    if (!file) {
      llvm::WithColor::error(llvm::errs())
          << "Failed to open: " << path << "\n";
      ++failures;
      continue;
    }
    sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sm, &context);
    if (!module) {
      llvm::WithColor::error(llvm::errs())
          << "Failed to parse: " << path << "\n";
      ++failures;
      continue;
    }

    mlir::OpPrintingFlags flags;
    flags.useLocalScope();
    module->print(llvm::outs(), flags);
    llvm::outs() << "\n";
    ++parsed;
  }

  if (ec) {
    llvm::WithColor::error(llvm::errs())
        << "Error reading directory: " << ec.message() << "\n";
    return 1;
  }

  llvm::outs() << "=== Summary: " << parsed << " file(s) parsed, " << failures
               << " failure(s) ===\n";
  return failures > 0 ? 1 : 0;
}
