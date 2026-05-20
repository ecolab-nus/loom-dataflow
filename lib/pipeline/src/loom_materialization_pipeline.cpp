/// Implementation of the Loom MLIR materialization pipeline.
///
/// Combines the Materialize pipeline and the One-Shot Bufferization pipeline
/// into a single in-memory pass manager run.
///
/// Provides a C++ API (runMaterializationPipeline) exposed via pybind11.

#include "loom_materialization_pipeline.h"
#include "Passes.h"
#include "Transforms/BufferizableOpInterfaceImpl.h"
// Forward-declare the tt-opt pass to avoid pulling in its full Passes.h
// (which re-includes GEN_PASS_REGISTRATION and causes redefinition conflicts).
// TODO: distinguish per-backend pass sets when multiple backends are supported.
namespace loom::passes {
std::unique_ptr<mlir::Pass> createConvertZeroFillLinalgMatmulToLoomPass();
std::unique_ptr<mlir::Pass> createFoldZeroFillLinalgPass();
} // namespace loom::passes

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "ADLDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"
#include "LoomDialect.h.inc"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

using namespace mlir;

namespace {

/// Parse the JSON block sizes string into a BlockSizeMap.
/// JSON format: {"func_name": {"SYM": value, ...}, ...}
/// Returns true on success and fills outMap; false on parse error (fills errMsg).
bool parseBlockSizesJson(const char *json_str,
                         loom::passes::BlockSizeMap &outMap,
                         std::string &errMsg) {
  if (!json_str || json_str[0] == '\0')
    return true; // empty → use placeholder solver

  auto parsed = llvm::json::parse(llvm::StringRef(json_str));
  if (!parsed) {
    errMsg = "Failed to parse block_sizes_json: " +
             llvm::toString(parsed.takeError());
    return false;
  }

  auto *root = parsed->getAsObject();
  if (!root) {
    errMsg = "block_sizes_json must be a JSON object at the top level";
    return false;
  }

  for (auto &[funcKey, symVals] : *root) {
    // UNSAT variant: null block sizes → skip (not added to outMap)
    if (symVals.kind() == llvm::json::Value::Null) {
      // llvm::errs() << "info: variant '" << funcKey
      //              << "' is UNSAT, will be omitted from output IR\n";
      continue;
    }
    auto *symObj = symVals.getAsObject();
    if (!symObj) {
      errMsg = "Value for key '" + funcKey.str() + "' must be a JSON object";
      return false;
    }
    llvm::StringMap<int64_t> symMap;
    for (auto &[symKey, val] : *symObj) {
      auto intVal = val.getAsInteger();
      if (!intVal) {
        errMsg = "Value for symbol '" + symKey.str() + "' in '" +
                 funcKey.str() + "' must be an integer";
        return false;
      }
      symMap[symKey] = *intVal;
    }
    outMap[funcKey] = std::move(symMap);
  }
  return true;
}

/// Core materialization pipeline logic.
std::pair<std::string, std::string>
runMaterializationCore(const char *input_mlir_text,
                       const char *block_sizes_json) {
  // --- Parse block sizes JSON ---
  loom::passes::BlockSizeMap blockSizeMap;
  std::string errMsg;
  bool hasExternalSizes =
      block_sizes_json && block_sizes_json[0] != '\0';

  if (hasExternalSizes) {
    if (!parseBlockSizesJson(block_sizes_json, blockSizeMap, errMsg))
      return {errMsg, ""};
  }

  // --- Set up MLIRContext with all required dialects ---
  MLIRContext context;

  DialectRegistry registry;
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerSubsetOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);

  context.loadDialect<loom::LoomDialect,
                      adl::ADLDialect,
                      func::FuncDialect,
                      arith::ArithDialect,
                      affine::AffineDialect,
                      tensor::TensorDialect,
                      linalg::LinalgDialect,
                      memref::MemRefDialect,
                      scf::SCFDialect,
                      bufferization::BufferizationDialect>();

  // Register Loom's BufferizableOpInterface models
  loom::registerBufferizableOpInterfaceExternalModels(&context);

  // --- Parse input MLIR from string ---
  auto inputBuf = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(input_mlir_text), "input_mlir");

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputBuf), llvm::SMLoc());
  auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    return {"Failed to parse input MLIR text", ""};

  // Expand special "ALL" binding to every candidate function in the input MLIR.
  // Explicit per-function bindings take precedence over "ALL".
  if (hasExternalSizes) {
    auto allIt = blockSizeMap.find("ALL");
    if (allIt != blockSizeMap.end()) {
      const auto allBinding = allIt->second;
      // Erase by key before inserting new entries. Inserting into StringMap may
      // rehash and invalidate iterators, so using `allIt` after insertions can
      // trigger LLVM StringMap internal assertions.
      blockSizeMap.erase("ALL");
      module->walk([&](func::FuncOp func) {
        StringRef funcName = func.getName();
        if (blockSizeMap.find(funcName) == blockSizeMap.end()) {
          blockSizeMap[funcName] = allBinding;
        }
      });
    }
  }

  // --- Build pass pipeline ---
  PassManager pm(&context);

  // Stage 1: Materialize symbolic block sizes
  if (hasExternalSizes) {
    pm.addPass(loom::passes::createMaterializePass(blockSizeMap));
  } else {
    pm.addPass(loom::passes::createMaterializePass());
  }

  // Stage 2: Canonicalize, remove dead symbols, bridge to OSB
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(loom::passes::createBridgeToOSBPass());

  // Stage 3: Lower affine with attributes (required before OSB)
  pm.addPass(loom::passes::createLowerAffineWithAttrPass());

  // Stage 4: One-Shot Bufferization
  bufferization::OneShotBufferizePassOptions osbOptions;
  osbOptions.allowUnknownOps = false;
  osbOptions.bufferizeFunctionBoundaries = true;
  osbOptions.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;

  pm.addPass(bufferization::createOneShotBufferizePass(osbOptions));
  pm.nest<ModuleOp>().addPass(
      bufferization::createOneShotBufferizePass(osbOptions));

  // Stage 5: Final cleanup (matches one_shot_bufferize single-stage driver)
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(loom::passes::createLowerLinalgCopyToLoomCopyPass());

  // Stage 6: Backend-specific TT optimizations
  // (matches tt-opt single-stage driver).
  // TODO: gate these passes on a backend enum once multiple backends are
  //       supported (e.g. TT-Metal vs. others).
  pm.addPass(loom::passes::createConvertZeroFillLinalgMatmulToLoomPass());
  pm.addPass(loom::passes::createFoldZeroFillLinalgPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // --- Run pipeline ---
  if (failed(pm.run(*module)))
    return {"Pipeline execution failed", ""};

  // --- Serialize output MLIR to string ---
  std::string output_mlir;
  llvm::raw_string_ostream outStream(output_mlir);

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  module->print(outStream, flags);
  outStream << "\n";

  return {"", output_mlir};
}

} // namespace

// ---------------------------------------------------------------------------
// C++ API
// ---------------------------------------------------------------------------

namespace loom {
namespace pipeline {

std::pair<std::string, std::string>
runMaterializationPipeline(const std::string &input_mlir_text,
                           const std::string &block_sizes_json) {
  return runMaterializationCore(input_mlir_text.c_str(),
                                block_sizes_json.c_str());
}

} // namespace pipeline
} // namespace loom
