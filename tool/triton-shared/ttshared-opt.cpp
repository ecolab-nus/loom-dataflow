// Combined driver: Triton-shared -> Affine pipeline.
//
// The driver expects a Triton `tt.shared` kernel together with a hardware
// description written in the (early) `df` dialect. It applies the two core
// passes—affinize, then grid-to-parallel—to expose the GPU launch grid as a
// single 3-D `affine.parallel`. Exploration then matches those induction
// variables against the declared hardware spatial dimensions, cloning the
// kernel for every viable binding. Each clone tracks which hardware dimension
// drives which loop through a `tmd.mapped_to` attribute, inserts `affine.for`
// nests when multiple "waves" are needed to time-multiplex the workload, and
// leaves the inner `scf.for` loops untouched to model per-core tile sequencing.
// The resulting module mirrors the structure of
// `test/Dialect/Triton/mm_fixed_strides/after_exploration.mlir`.
//
// Usage:
//   tmd_triton_shared_to_affine --ttshared <ttshared.mlir> --df <df.mlir>
//
#include "reinterpret_cast_reuse.h"
#include "spatial_mapping.h"
#include "triton_shared_affinize.h"
#include "triton_shared_grid_to_parallel.h"

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

using namespace mlir;

static llvm::cl::opt<std::string>
    clTTSharedInput("ttshared", llvm::cl::desc("Path to ttshared MLIR file"),
                    llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    clDfInput("df",
              llvm::cl::desc("Path to DF MLIR file (spatial description)"),
              llvm::cl::value_desc("filename"), llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TMD Triton-shared to Affine pipeline\n");

  // Setup context and register required dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                  mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::bufferization::BufferizationDialect,
                  tmd::df::DataflowDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Parse DF module for spatial dimensions.
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

  llvm::SmallVector<tmd_affine::SpatialDimInfo, 8> spatialDims;
  if (failed(tmd_affine::collectSpatialDims(*dfModule, spatialDims))) {
    llvm::WithColor::error(llvm::errs())
        << "No df.spatial_dim found or parse failure in DF module\n";
    return 1;
  }

  // Parse the Triton-shared module to transform.
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

  // Build and run the pipeline: affinize -> grid_to_parallel.
  PassManager pm(&context);
  pm.addPass(tmd::passes::createTritonSharedAffinizePass());
  pm.addPass(tmd::passes::createTritonSharedGridToParallelPass());
  if (failed(pm.run(*tsModule))) {
    llvm::WithColor::error(llvm::errs())
        << "Pipeline (affinize -> grid_to_parallel) failed\n";
    return 2;
  }

  // Now explore spatial mappings over the outermost affine.parallel with DF.
  OwningOpRef<ModuleOp> explored =
      tmd_affine::enumerateSpatialMappingsWithOuterFors(*tsModule, spatialDims);

  // Merge DF declarations and generated clones into a single module.
  OwningOpRef<ModuleOp> merged = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(merged->getBodyRegion());
  IRMapping mapping;
  for (Operation &op : *dfModule->getBody())
    builder.clone(op, mapping);
  for (Operation &op : *explored->getBody())
    builder.clone(op, mapping);

  // Annotate reinterpret_cast ops with reuse information relative to the
  // surrounding spatial/temporal iterators.
  PassManager annotatePM(&context);
  annotatePM.addPass(tmd::passes::createAnnotateReinterpretCastReusePass());
  if (failed(annotatePM.run(*merged))) {
    llvm::WithColor::error(llvm::errs()) << "Reuse annotation pass failed\n";
    return 3;
  }

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  merged->print(llvm::outs(), flags);
  llvm::outs() << "\n";
  return 0;
}
