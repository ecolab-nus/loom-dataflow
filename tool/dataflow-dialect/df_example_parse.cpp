#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

#include "DataflowDialect.h.inc"

using namespace mlir;

int main(int argc, char **argv) {
  MLIRContext context;
  context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::affine::AffineDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<tmd::df::DataflowDialect>();

  const char *filename = nullptr;
  static std::string defaultPath;
  if (argc > 1) {
    filename = argv[1];
  } else {
    llvm::SmallString<256> execPath(argv[0]);
    if (std::error_code ec = llvm::sys::fs::real_path(execPath, execPath))
      (void)ec; // Best-effort; fall back to argv[0] path if this fails.

    llvm::StringRef execDir = llvm::sys::path::parent_path(execPath);
    llvm::StringRef buildDir = execDir;
    llvm::StringRef projectRoot = llvm::sys::path::parent_path(buildDir);

    llvm::SmallString<256> candidate(projectRoot);
    llvm::sys::path::append(candidate, "test", "Dialect", "DataflowDialect",
                            "2D_mesh.mlir");

    if (!candidate.empty() && llvm::sys::fs::exists(candidate)) {
      defaultPath = candidate.str().str();
    } else {
      // Fall back to a relative path; this allows running from the repository
      // root or build directory when the file exists in the default location.
      defaultPath = "test/Dialect/DataflowDialect/2D_mesh.mlir";
    }

    filename = defaultPath.c_str();
  }

  auto file = mlir::openInputFile(filename);
  if (!file) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open input file: " << filename << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs()) << "Failed to parse MLIR module\n";
    return 1;
  }

  AsmState state(*module);
  module->print(llvm::outs(), state);
  llvm::outs() << "\n";
  return 0;
}
