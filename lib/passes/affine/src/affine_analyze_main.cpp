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
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;

// Dataflow dialect for parsing df module sections.
#include "DataflowDialect.h.inc"
#include "DataflowOps.h.inc"

namespace loom_affine_analysis {

LogicalResult runSyntaxCheck(func::FuncOp funcOp);
void attachPrimitiveReuseVectors(func::FuncOp funcOp);
void annotateSpatialInvariance(func::FuncOp funcOp);

} // namespace loom_affine_analysis

int main(int argc, char **argv) {
  MLIRContext context;
  (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::func::FuncDialect, mlir::affine::AffineDialect,
                      mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
  context.loadDialect<loom::df::DataflowDialect>();

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

  // No SSA name preservation needed since we only print a text report.

  bool hadError = false;
  module->walk([&](func::FuncOp funcOp) {
    if (failed(loom_affine_analysis::runSyntaxCheck(funcOp))) {
      hadError = true;
      return;
    }
    loom_affine_analysis::attachPrimitiveReuseVectors(funcOp);
    loom_affine_analysis::annotateSpatialInvariance(funcOp);
  });

  if (hadError)
    return 2;
  // Print the module so that attached attributes are visible in IR.
  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
