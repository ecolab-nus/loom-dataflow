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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

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
  tmd_affine::HardwareInfo hardwareInfo;
  if (failed(tmd_affine::GetHardwareInfoForExploration(*dfModule, hardwareInfo))) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to collect hardware information from DF module\n";
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
      tmd_affine::EnumerateSpatialMappings(*affineModule, hardwareInfo);

  // Merge DF declarations and generated Affine clones into a single module to
  // avoid duplicate alias ids and produce a single well-formed module.
  OwningOpRef<ModuleOp> merged = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(merged->getBodyRegion());
  IRMapping mapping;
  for (Operation &op : *dfModule->getBody())
    builder.clone(op, mapping);
  for (Operation &op : *out->getBody())
    builder.clone(op, mapping);

  AsmState mergedState(*merged);
  merged->print(llvm::outs(), mergedState);
  llvm::outs() << "\n";
  return 0;
}
