/// Implementation of the Loom MLIR materialization pipeline.
///
/// Combines the Materialize pipeline (canonicalize_main.cpp) and the
/// One-Shot Bufferization pipeline (one_shot_bufferize_main.cpp) into a
/// single in-memory pass manager run.
///
/// Provides both a C++ API (runMaterializationPipeline) and a legacy
/// C API (loom_run_full_pipeline) for backward compatibility.

#include "loom_materialization_pipeline.h"
#include "loom_pipeline_api.h"
#include "Passes.h"
#include "Transforms/BufferizableOpInterfaceImpl.h"

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
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "DataflowDialect.h.inc"
#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <string>

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
      llvm::errs() << "info: variant '" << funcKey
                   << "' is UNSAT, will be omitted from output IR\n";
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

/// Core materialization pipeline logic shared by C++ and C APIs.
std::string runMaterializationCore(const char *input_mlir_path,
                                   const char *block_sizes_json,
                                   const char *output_mlir_path) {
  // --- Parse block sizes JSON ---
  loom::passes::BlockSizeMap blockSizeMap;
  std::string errMsg;
  bool hasExternalSizes =
      block_sizes_json && block_sizes_json[0] != '\0';

  if (hasExternalSizes) {
    if (!parseBlockSizesJson(block_sizes_json, blockSizeMap, errMsg))
      return errMsg;
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
  context.appendDialectRegistry(registry);

  context.loadDialect<loom::LoomDialect,
                      loom::df::DataflowDialect,
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

  // --- Parse input MLIR ---
  auto fileOrErr = llvm::MemoryBuffer::getFile(input_mlir_path);
  if (std::error_code ec = fileOrErr.getError())
    return std::string("Could not open input file '") + input_mlir_path +
           "': " + ec.message();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    return std::string("Failed to parse MLIR file: ") + input_mlir_path;

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

  // Stage 5: Final cleanup
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // --- Run pipeline ---
  if (failed(pm.run(*module)))
    return "Pipeline execution failed";

  // --- Write output MLIR ---
  std::error_code ec;
  llvm::raw_fd_ostream outStream(output_mlir_path, ec);
  if (ec)
    return std::string("Could not open output file '") + output_mlir_path +
           "': " + ec.message();

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  module->print(outStream, flags);
  outStream << "\n";

  return "";
}

} // namespace

// ---------------------------------------------------------------------------
// C++ API
// ---------------------------------------------------------------------------

namespace loom {
namespace pipeline {

std::string runMaterializationPipeline(const std::string &input_mlir_path,
                                       const std::string &block_sizes_json,
                                       const std::string &output_mlir_path) {
  return runMaterializationCore(input_mlir_path.c_str(),
                                block_sizes_json.c_str(),
                                output_mlir_path.c_str());
}

} // namespace pipeline
} // namespace loom

// ---------------------------------------------------------------------------
// Legacy C API (backward compatibility with ctypes-based orchestrator)
// ---------------------------------------------------------------------------

extern "C" {

__attribute__((visibility("default")))
int loom_run_full_pipeline(const char *input_mlir_path,
                           const char *block_sizes_json,
                           const char *output_mlir_path,
                           char **error_msg) {
  if (error_msg)
    *error_msg = nullptr;

  std::string err = runMaterializationCore(input_mlir_path, block_sizes_json,
                                           output_mlir_path);
  if (!err.empty()) {
    if (error_msg)
      *error_msg = strdup(err.c_str());
    return 1;
  }
  return 0;
}

__attribute__((visibility("default")))
void loom_free_string(char *str) {
  free(str);
}

} // extern "C"
