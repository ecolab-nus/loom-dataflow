#include "affine_tile.h"
// Minimal test driver that parses a module, runs the tiling pass, and prints.
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include <string>

using namespace mlir;

/**
 * Test driver: reads an MLIR module from stdin or a file, applies the tiling
 * transform to every function using the provided tiling factor (argv[2],
 * default 8), and prints the transformed module.
 *
 * Usage:
 *   loom_affine_tile <input.mlir> [tiling_factor]
 */
int main(int argc, char **argv) {
  MLIRContext context;
  (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::func::FuncDialect, mlir::affine::AffineDialect,
                      mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();

  llvm::SourceMgr sourceMgr;
  const char *filename = argc > 1 ? argv[1] : "-";
  auto file = mlir::openInputFile(filename);
  if (!file) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open input file: " << filename << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs()) << "Failed to parse MLIR module\n";
    return 1;
  }

  // Parse optional tiling factor (argv[2], default 8) and tile dim index
  // (argv[3], default 0).
  int64_t tilingFactor = 8;
  unsigned tileDimIndex = 0;
  if (argc > 2) {
    std::string s(argv[2]);
    try {
      size_t idx = 0;
      long v = std::stol(s, &idx, 10);
      if (idx == s.size())
        tilingFactor = static_cast<int64_t>(v);
    } catch (...) {
      // ignore, keep default
    }
  }
  if (argc > 3) {
    std::string s(argv[3]);
    try {
      size_t idx = 0;
      long v = std::stol(s, &idx, 10);
      if (idx == s.size() && v >= 0)
        tileDimIndex = static_cast<unsigned>(v);
    } catch (...) {
      // ignore, keep default
    }
  }

  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(
      loom_affine::createAffineTilePass(tilingFactor, tileDimIndex));
  if (failed(pm.run(*module))) {
    llvm::WithColor::error(llvm::errs()) << "Tiling failed\n";
    return 2;
  }

  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
