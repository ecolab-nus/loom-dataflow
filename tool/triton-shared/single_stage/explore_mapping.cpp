// Driver for enumerating Triton-shared grid-to-spatial mappings and printing a
// combined module.
//
// Usage:
//   tmd_triton_shared_explore --ttshared <ttshared.mlir> --df <df.mlir>
//   [--grid-dims N]
//
// The driver loads both modules, collects spatial dimensions from the DF
// module, enumerates all unique assignments of grid dimensions {x,y,z} to the
// hardware spatial dimensions, and prints a new module containing one clone of
// each function per mapping with attributes encoding the binding. When the
// hardware mesh cannot cover the full grid in one shot, the explorer also
// inserts outer `affine.for` loops to model sequential "waves" while leaving
// the inner `scf.for` loops to represent per-core tile sequencing.

#include "reinterpret_cast_reuse.h"
#include "spatial_mapping.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

#include "DataflowDialect.h.inc"
#include "DataflowOps.h.inc"

using namespace mlir;

static llvm::cl::opt<std::string>
    clTTSharedInput("ttshared", llvm::cl::desc("Path to ttshared MLIR file"),
                    llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    clDfInput("df",
              llvm::cl::desc("Path to DF MLIR file (spatial description)"),
              llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<unsigned> clNumGridDims(
    "grid-dims", llvm::cl::desc("Number of grid dimensions to consider (1..3)"),
    llvm::cl::init(3));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TMD Triton-shared spatial explorer\n");

  // Legacy option retained for compatibility of CLI; ignored in the new format.
  (void)clNumGridDims;

  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                  mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::bufferization::BufferizationDialect,
                  tmd::df::DataflowDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

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

  // Parse the ttshared module to explore.
  llvm::SourceMgr tsSm;
  auto tsFile = mlir::openInputFile(clTTSharedInput);
  if (!tsFile) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open ttshared input file: " << clTTSharedInput << "\n";
    return 1;
  }
  tsSm.AddNewSourceBuffer(std::move(tsFile), llvm::SMLoc());
  OwningOpRef<ModuleOp> tsModule = parseSourceFile<ModuleOp>(tsSm, &context);
  if (!tsModule) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to parse ttshared MLIR module\n";
    return 1;
  }

  // Enumerate all grid-to-spatial assignments for the ttshared module.
  // New format: enumerate mappings directly over the outermost affine.parallel
  // in the Triton-shared-after-grid-to-parallel module. We ignore numGridDims
  // and rely purely on the number of iterators in the parallel op.
  // Enumerate mappings and also explore outer-for loop orderings.
  OwningOpRef<ModuleOp> out =
      tmd_affine::enumerateSpatialMappingsWithOuterFors(*tsModule, spatialDims);

  // Merge DF declarations and generated clones into a single module.
  OwningOpRef<ModuleOp> merged = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(merged->getBodyRegion());
  IRMapping mapping;
  for (Operation &op : *dfModule->getBody())
    builder.clone(op, mapping);
  for (Operation &op : *out->getBody())
    builder.clone(op, mapping);

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  merged->print(llvm::outs(), flags);
  llvm::outs() << "\n";
  return 0;
}
