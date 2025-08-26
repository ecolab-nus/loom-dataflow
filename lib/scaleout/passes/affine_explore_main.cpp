// Driver for enumerating spatial mappings and printing a combined module.
//
// Usage:
//   tmd_affine_explore --affine <affine.mlir> --df <df.mlir>
//
// The driver loads both modules, collects spatial dimensions from the DF
// module, enumerates all unique mappings of these dimensions to the iterators
// of each function's first outermost `affine.parallel`, and prints a new
// module containing a clone of the function for each mapping. Each created
// inner loop is annotated with `tmd.mapped_to` and function names are suffixed
// to encode the mapping.
#include "spatial_mapping.h"

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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

#include "DataflowDialect.h.inc"
#include "DataflowOps.h.inc"

using namespace mlir;

static llvm::cl::opt<std::string>
    clAffineInput("affine", llvm::cl::desc("Path to Affine MLIR file"),
                  llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    clDfInput("df",
              llvm::cl::desc("Path to DF MLIR file (spatial description)"),
              llvm::cl::value_desc("filename"), llvm::cl::Required);

// No tile-dim flag: we will enumerate all possible mappings.

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TMD Affine spatial explorer\n");

  MLIRContext context;
  (void)context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::affine::AffineDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<tmd::df::DataflowDialect>();

  // Parse DF module containing spatial dimensions.
  llvm::SourceMgr dfSm;
  auto dfFile = mlir::openInputFile(clDfInput);
  if (!dfFile) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open DF input file: " << clDfInput << "\n";
    return 1;
  }
  dfSm.AddNewSourceBuffer(std::move(dfFile), llvm::SMLoc());
  OwningOpRef<ModuleOp> dfModule = parseSourceFile<ModuleOp>(dfSm, &context);
  if (!dfModule) {
    llvm::WithColor::error(llvm::errs()) << "Failed to parse DF MLIR module\n";
    return 1;
  }

  // Collect spatial dimensions.
  llvm::SmallVector<tmd_affine::SpatialDimInfo, 8> spatialDims;
  if (failed(tmd_affine::collectSpatialDims(*dfModule, spatialDims))) {
    llvm::WithColor::error(llvm::errs())
        << "No df.spatial_dim found or parse failure in DF module\n";
    return 1;
  }

  // Parse the Affine module to transform.
  llvm::SourceMgr affineSm;
  auto affineFile = mlir::openInputFile(clAffineInput);
  if (!affineFile) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open Affine input file: " << clAffineInput << "\n";
    return 1;
  }
  affineSm.AddNewSourceBuffer(std::move(affineFile), llvm::SMLoc());
  OwningOpRef<ModuleOp> affineModule =
      parseSourceFile<ModuleOp>(affineSm, &context);
  if (!affineModule) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to parse Affine MLIR module\n";
    return 1;
  }

  // Enumerate all mapping combinations for the Affine module.
  OwningOpRef<ModuleOp> out =
      tmd_affine::enumerateSpatialMappings(*affineModule, spatialDims);

  // Print the DF (hardware description) module first, then the generated Affine
  // module. This yields two top-level modules in the output stream, e.g.:
  //   module { ...hardware... }
  //   module { ...generated clones... }
  AsmState dfAsmState(*dfModule);
  dfModule->print(llvm::outs(), dfAsmState);
  llvm::outs() << "\n";

  AsmState outAsmState(*out);
  out->print(llvm::outs(), outAsmState);
  llvm::outs() << "\n";
  return 0;
}
