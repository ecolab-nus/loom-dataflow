// Combined driver: Triton-shared -> Affine pipeline.
//
// The driver expects a Triton `tt.shared` kernel together with a hardware
// description written in the (early) `df` dialect. It applies the two core
// passes—affinize, then grid-to-parallel—to expose the GPU launch grid as a
// single 3-D `affine.parallel`. Exploration then matches those induction
// variables against the declared hardware spatial dimensions, cloning the
// kernel for every viable binding. Each clone tracks which hardware dimension
// drives which loop through a `loom.mapped_to` attribute, inserts `affine.for`
// nests when multiple "waves" are needed to time-multiplex the workload, and
// leaves the inner `scf.for` loops untouched to model per-core tile sequencing.
// The resulting module mirrors the structure of
// `test/Dialect/Triton/mm_fixed_strides/after_exploration.mlir`.
//
// Usage:
//   ttshared-opt --ttshared <ttshared.mlir> --df <df.mlir>
//
#include "Passes.h"
#include "hardware_info.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

#include "DataflowDialect.h.inc"
#include "DataflowOps.h.inc"
#include "LoomDialect.h.inc"

using namespace mlir;

static llvm::cl::opt<std::string>
    clTTSharedInput("ttshared", llvm::cl::desc("Path to ttshared MLIR file"),
                    llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    clDfInput("df",
              llvm::cl::desc("Path to DF MLIR file (spatial description)"),
              llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<bool> clMapAnalysisOnly(
    "map-analysis-only",
    llvm::cl::desc("Only attach loom.copy.candidates; do not clone functions"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    clSkipTileScfForToL1("skip-tile-scf-for-to-l1",
                         llvm::cl::desc("Skip the tile-scf-for-to-l1 pass"),
                         llvm::cl::init(true));

static llvm::cl::opt<bool>
    clDumpIntermediate("dump-intermediate",
                       llvm::cl::desc("Dump MLIR after each pass to stderr"),
                       llvm::cl::init(false));

static llvm::cl::opt<std::string> clDumpDir(
    "dump-dir",
    llvm::cl::desc("Directory to dump intermediate MLIR files (optional)"),
    llvm::cl::value_desc("dir"), llvm::cl::init(""));

// No max-variants flag; we enumerate all combinations.

/// Dump the given module to a file within 'dirPath' with the specified
/// 'fileName'. Creates the directory if it doesn't exist.
static LogicalResult dumpModuleToFile(ModuleOp module, StringRef dirPath,
                                      StringRef fileName) {
  if (dirPath.empty())
    return success();

  // Ensure the directory exists.
  if (std::error_code ec = llvm::sys::fs::create_directories(dirPath)) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to create dump directory '" << dirPath
        << "': " << ec.message() << "\n";
    return failure();
  }

  llvm::SmallString<256> fullPath(dirPath);
  llvm::sys::path::append(fullPath, fileName);

  std::string errMsg;
  auto out = mlir::openOutputFile(fullPath, &errMsg);
  if (!out) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open dump file '" << fullPath << "': " << errMsg << "\n";
    return failure();
  }

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  module->print(out->os(), flags);
  out->os() << "\n";
  out->keep();
  return success();
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "LOOM Triton-shared to Affine pipeline\n");

  // Setup context and register required dialects.
  mlir::DialectRegistry registry;
  // Register all dialects and external models (e.g., BufferizableOpInterface
  // implementations for Linalg/Tensor/SCF/etc.), then ensure our DF dialect is
  // present.
  mlir::registerAllDialects(registry);
  registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                  mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::bufferization::BufferizationDialect,
                  loom::df::DataflowDialect, loom::LoomDialect>();
  MLIRContext context(registry, clDumpIntermediate
                                    ? MLIRContext::Threading::DISABLED
                                    : MLIRContext::Threading::ENABLED);
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

  loom::HardwareInfo hardwareInfo;
  if (failed(loom::GetHardwareInfoForExploration(*dfModule, hardwareInfo))) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to collect hardware information from DF module\n";
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

  // Run affinization, then dump if requested.
  PassManager pmAff(&context);
  if (clDumpIntermediate) {
    pmAff.enableIRPrinting(
        [](mlir::Pass *, mlir::Operation *) { return false; },
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false, llvm::errs());
  }
  pmAff.addPass(loom::passes::createTritonSharedAffinizePass());
  if (failed(pmAff.run(*tsModule))) {
    llvm::WithColor::error(llvm::errs()) << "Affinization pass failed\n";
    return 2;
  }
  {
    PassManager cleanupPM(&context);
    if (clDumpIntermediate) {
      cleanupPM.enableIRPrinting(
          [](mlir::Pass *, mlir::Operation *) { return false; },
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/false, llvm::errs());
    }
    cleanupPM.addPass(loom::passes::createConstDedupCleanupPass());
    if (failed(cleanupPM.run(*tsModule))) {
      llvm::WithColor::error(llvm::errs()) << "Constant cleanup pass failed\n";
      return 7;
    }
  }
  if (!clDumpDir.empty() &&
      failed(
          dumpModuleToFile(*tsModule, clDumpDir, "01_after_affinization.mlir")))
    return 5;

  // Run grid-to-parallel, then dump if requested.
  PassManager pmGrid(&context);
  if (clDumpIntermediate) {
    pmGrid.enableIRPrinting(
        [](mlir::Pass *, mlir::Operation *) { return false; },
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false, llvm::errs());
  }
  pmGrid.addPass(loom::passes::createTritonSharedGridToParallelPass());
  if (failed(pmGrid.run(*tsModule))) {
    llvm::WithColor::error(llvm::errs()) << "Grid-to-parallel pass failed\n";
    return 2;
  }
  {
    PassManager cleanupPM(&context);
    if (clDumpIntermediate) {
      cleanupPM.enableIRPrinting(
          [](mlir::Pass *, mlir::Operation *) { return false; },
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/false, llvm::errs());
    }
    cleanupPM.addPass(loom::passes::createConstDedupCleanupPass());
    if (failed(cleanupPM.run(*tsModule))) {
      llvm::WithColor::error(llvm::errs()) << "Constant cleanup pass failed\n";
      return 7;
    }
  }
  if (!clDumpDir.empty() &&
      failed(dumpModuleToFile(*tsModule, clDumpDir,
                              "02_after_grid_to_parallel.mlir")))
    return 5;

  // Merge DF decls into the transformed tsModule so passes can see DF.
  {
    OpBuilder mb(tsModule->getBodyRegion());
    IRMapping map;
    for (Operation &op : *dfModule->getBody())
      mb.clone(op, map);
  }

  // Run spatial mapping as a pass (with outer-for permutations).
  PassManager spatialPM(&context);
  if (clDumpIntermediate) {
    spatialPM.enableIRPrinting(
        [](mlir::Pass *, mlir::Operation *) { return false; },
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false, llvm::errs());
  }
  spatialPM.addPass(loom::passes::createTritonSharedExploreSpatialMappingsPass(
      /*withOuterFors=*/true));
  if (failed(spatialPM.run(*tsModule))) {
    llvm::WithColor::error(llvm::errs())
        << "Spatial mapping exploration pass failed\n";
    return 3;
  }

  // Fallback: if no functions are present (unexpected), enumerate directly.
  bool hasFuncs = false;
  for (Operation &op : *tsModule->getBody())
    if (isa<func::FuncOp>(&op)) {
      hasFuncs = true;
      break;
    }
  if (!hasFuncs) {
    OwningOpRef<ModuleOp> explored =
        loom::EnumerateSpatialMappings(*tsModule, hardwareInfo);
    // Replace non-DF ops with explored clones.
    SmallVector<Operation *, 16> toErase;
    for (Operation &op : *tsModule->getBody()) {
      Dialect *dialect = op.getDialect();
      if (dialect && dialect->getNamespace() == StringRef("df"))
        continue;
      toErase.push_back(&op);
    }
    for (auto it = toErase.rbegin(); it != toErase.rend(); ++it)
      (*it)->erase();
    OpBuilder rb(tsModule->getBodyRegion());
    IRMapping m;
    for (Operation &op : *explored->getBody())
      rb.clone(op, m);
  }
  {
    PassManager cleanupPM(&context);
    if (clDumpIntermediate) {
      cleanupPM.enableIRPrinting(
          [](mlir::Pass *, mlir::Operation *) { return false; },
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/false, llvm::errs());
    }
    cleanupPM.addPass(loom::passes::createConstDedupCleanupPass());
    if (failed(cleanupPM.run(*tsModule))) {
      llvm::WithColor::error(llvm::errs()) << "Constant cleanup pass failed\n";
      return 7;
    }
  }
  if (!clDumpDir.empty() &&
      failed(
          dumpModuleToFile(*tsModule, clDumpDir, "03_after_exploration.mlir")))
    return 5;

  // Run hoist block loading pass.
  PassManager hoistPM(&context);
  if (clDumpIntermediate) {
    hoistPM.enableIRPrinting(
        [](mlir::Pass *, mlir::Operation *) { return false; },
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false, llvm::errs());
  }
  hoistPM.addPass(loom::passes::createHoistBlockLoadingPass());
  if (failed(hoistPM.run(*tsModule))) {
    llvm::WithColor::error(llvm::errs()) << "Hoist block loading pass failed\n";
    return 9;
  }
  {
    PassManager cleanupPM(&context);
    if (clDumpIntermediate) {
      cleanupPM.enableIRPrinting(
          [](mlir::Pass *, mlir::Operation *) { return false; },
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/false, llvm::errs());
    }
    cleanupPM.addPass(loom::passes::createConstDedupCleanupPass());
    if (failed(cleanupPM.run(*tsModule))) {
      llvm::WithColor::error(llvm::errs()) << "Constant cleanup pass failed\n";
      return 7;
    }
  }
  if (!clDumpDir.empty() &&
      failed(dumpModuleToFile(*tsModule, clDumpDir,
                              "04_after_hoist_block_loading.mlir")))
    return 5;

  // Annotate reinterpret_cast ops with reuse information.
  PassManager annotatePM(&context);
  if (clDumpIntermediate) {
    annotatePM.enableIRPrinting(
        [](mlir::Pass *, mlir::Operation *) { return false; },
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false, llvm::errs());
  }
  annotatePM.addPass(loom::passes::createAnnotateReinterpretCastReusePass());
  if (failed(annotatePM.run(*tsModule))) {
    llvm::WithColor::error(llvm::errs()) << "Reuse annotation pass failed\n";
    return 3;
  }
  {
    PassManager cleanupPM(&context);
    if (clDumpIntermediate) {
      cleanupPM.enableIRPrinting(
          [](mlir::Pass *, mlir::Operation *) { return false; },
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/false, llvm::errs());
    }
    cleanupPM.addPass(loom::passes::createConstDedupCleanupPass());
    if (failed(cleanupPM.run(*tsModule))) {
      llvm::WithColor::error(llvm::errs()) << "Constant cleanup pass failed\n";
      return 7;
    }
  }
  if (!clDumpDir.empty() &&
      failed(dumpModuleToFile(*tsModule, clDumpDir,
                              "05_after_reuse_annotation.mlir")))
    return 5;

  // Enumerate copy interconnect broadcast choices.
  PassManager mappingPM(&context);
  if (clDumpIntermediate) {
    mappingPM.enableIRPrinting(
        [](mlir::Pass *, mlir::Operation *) { return false; },
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false, llvm::errs());
  }
  mappingPM.addPass(loom::passes::createEnumerateCopyBroadcastPass(
      /*analysisOnly=*/clMapAnalysisOnly));
  if (failed(mappingPM.run(*tsModule))) {
    llvm::WithColor::error(llvm::errs())
        << "Alloc/Copy mapping exploration failed\n";
    return 4;
  }
  {
    PassManager cleanupPM(&context);
    if (clDumpIntermediate) {
      cleanupPM.enableIRPrinting(
          [](mlir::Pass *, mlir::Operation *) { return false; },
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/false, llvm::errs());
    }
    cleanupPM.addPass(loom::passes::createConstDedupCleanupPass());
    if (failed(cleanupPM.run(*tsModule))) {
      llvm::WithColor::error(llvm::errs()) << "Constant cleanup pass failed\n";
      return 7;
    }
  }
  if (!clDumpDir.empty() &&
      failed(dumpModuleToFile(*tsModule, clDumpDir,
                              "06_after_memref_mapping.mlir")))
    return 5;

  // Canonicalize: Materialize, staticize types, lower loom ops, and affinize.
  PassManager canonicalizePM(&context);
  if (clDumpIntermediate) {
    canonicalizePM.enableIRPrinting(
        [](mlir::Pass *, mlir::Operation *) { return false; },
        [](mlir::Pass *, mlir::Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false, llvm::errs());
  }
  /// Step 1: Materialize - Replace loom.get_module_attribute with
  /// arith.constant.
  canonicalizePM.addPass(loom::passes::createMaterializePass());
  /// Step 2: Staticize - Convert dynamic memref/tensor types to static types.
  canonicalizePM.addPass(loom::passes::createStaticizeTypesPass());
  /// Step 3: Lower - Lower loom operations to memref dialect.
  canonicalizePM.addPass(loom::passes::createLoomToMemRefLoweringPass());
  /// Step 4: Affinize - Convert index arithmetic to affine IR.
  canonicalizePM.addPass(loom::passes::createTritonSharedAffinizePass());
  if (failed(canonicalizePM.run(*tsModule))) {
    llvm::WithColor::error(llvm::errs()) << "Canonicalize pass failed\n";
    return 2;
  }
  {
    PassManager cleanupPM(&context);
    if (clDumpIntermediate) {
      cleanupPM.enableIRPrinting(
          [](mlir::Pass *, mlir::Operation *) { return false; },
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/false, llvm::errs());
    }
    cleanupPM.addPass(loom::passes::createConstDedupCleanupPass());
    if (failed(cleanupPM.run(*tsModule))) {
      llvm::WithColor::error(llvm::errs()) << "Constant cleanup pass failed\n";
      return 7;
    }
  }
  if (!clDumpDir.empty() &&
      failed(dumpModuleToFile(*tsModule, clDumpDir,
                              "06_after_canonicalization.mlir")))
    return 5;

  // Run One-Shot Bufferize to convert tensors to memrefs, while allowing
  // unknown ops (e.g., df.*) to be preserved.
  {
    PassManager bufPM(&context);
    if (clDumpIntermediate) {
      bufPM.enableIRPrinting(
          [](mlir::Pass *, mlir::Operation *) { return false; },
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/false, llvm::errs());
    }
    mlir::bufferization::OneShotBufferizePassOptions bopts;
    bopts.allowUnknownOps = true;
    bopts.allowReturnAllocsFromLoops = true;
    bufPM.addPass(mlir::bufferization::createOneShotBufferizePass(bopts));
    if (failed(bufPM.run(*tsModule))) {
      llvm::WithColor::error(llvm::errs())
          << "One-Shot Bufferize pass failed\n";
      return 8;
    }
  }
  {
    PassManager cleanupPM(&context);
    if (clDumpIntermediate) {
      cleanupPM.enableIRPrinting(
          [](mlir::Pass *, mlir::Operation *) { return false; },
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/false, llvm::errs());
    }
    cleanupPM.addPass(loom::passes::createConstDedupCleanupPass());
    if (failed(cleanupPM.run(*tsModule))) {
      llvm::WithColor::error(llvm::errs()) << "Constant cleanup pass failed\n";
      return 7;
    }
  }

  if (!clDumpDir.empty() &&
      failed(dumpModuleToFile(*tsModule, clDumpDir,
                              "07_after_bufferization.mlir")))
    return 5;

  // Tile scf.for loops to fit L1, then dump if requested.
  if (!clSkipTileScfForToL1) {
    PassManager tilePM(&context);
    if (clDumpIntermediate) {
      tilePM.enableIRPrinting(
          [](mlir::Pass *, mlir::Operation *) { return false; },
          [](mlir::Pass *, mlir::Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/false,
          /*printAfterOnlyOnFailure=*/false, llvm::errs());
    }
    tilePM.addPass(loom::passes::createTileScfForToL1Pass());
    if (failed(tilePM.run(*tsModule))) {
      llvm::WithColor::error(llvm::errs()) << "SCF tiling to L1 pass failed\n";
      return 6;
    }

    // Cleanup after tiling pass.
    {
      PassManager cleanupPM(&context);
      if (clDumpIntermediate) {
        cleanupPM.enableIRPrinting(
            [](mlir::Pass *, mlir::Operation *) { return false; },
            [](mlir::Pass *, mlir::Operation *) { return true; },
            /*printModuleScope=*/true,
            /*printAfterOnlyOnChange=*/false,
            /*printAfterOnlyOnFailure=*/false, llvm::errs());
      }
      cleanupPM.addPass(loom::passes::createConstDedupCleanupPass());
      if (failed(cleanupPM.run(*tsModule))) {
        llvm::WithColor::error(llvm::errs())
            << "Constant cleanup pass failed\n";
        return 7;
      }
    }

    // Dump after tiling only if tiling pass was executed.
    if (!clDumpDir.empty() &&
        failed(
            dumpModuleToFile(*tsModule, clDumpDir, "08_after_for_tiling.mlir")))
      return 5;
  }

  // (Old merged-based path removed)

  // Reorder DF ops to the front (stable order).
  {
    Block &body = *tsModule->getBody();
    SmallVector<Operation *, 16> dfOps;
    for (Operation &op : body) {
      Dialect *dialect = op.getDialect();
      if (dialect && dialect->getNamespace() == StringRef("df"))
        dfOps.push_back(&op);
    }
    for (Operation *op : llvm::reverse(dfOps))
      op->moveBefore(&body.front());
  }

  // Print to stdout only if no dump directory was specified.
  if (clDumpDir.empty()) {
    mlir::OpPrintingFlags flags;
    flags.useLocalScope();
    tsModule->print(llvm::outs(), flags);
    llvm::outs() << "\n";
  }

  return 0;
}
