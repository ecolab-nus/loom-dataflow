/// Implementation of the Loom MLIR materialization pipeline.
///
/// Combines the Materialize pipeline and the One-Shot Bufferization pipeline
/// into a single in-memory pass manager run.
///
/// Provides a C++ API (runMaterializationPipeline) exposed via pybind11.

#include "loom_materialization_pipeline.h"
#include "Passes.h"
#include "Transforms/BufferizableOpInterfaceImpl.h"
#include "hw_op_registry.h"
#include "target.h"
// Forward-declare the tt-opt pass to avoid pulling in its full Passes.h
// (which re-includes GEN_PASS_REGISTRATION and causes redefinition conflicts).
// TODO: distinguish per-backend pass sets when multiple backends are supported.
namespace loom::passes {
std::unique_ptr<mlir::Pass> createConvertZeroFillLinalgMatmulToLoomPass();
std::unique_ptr<mlir::Pass> createFoldZeroFillLinalgPass();
std::unique_ptr<mlir::Pass> createSplitBinaryScalarChainPass();
} // namespace loom::passes

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

#include "ADL/IR/ADLDialect.h"
#include "ADL/IR/ADLTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "ADL/IR/ADLOps.h"
#include "LoomDialect.h.inc"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace {

std::vector<llvm::StringMap<int64_t>> cloneBindingList(
    const std::vector<llvm::StringMap<int64_t>> &bindings) {
  std::vector<llvm::StringMap<int64_t>> cloned;
  cloned.reserve(bindings.size());
  for (const auto &binding : bindings) {
    llvm::StringMap<int64_t> bindingClone;
    for (const auto &entry : binding)
      bindingClone[entry.first()] = entry.second;
    cloned.push_back(std::move(bindingClone));
  }
  return cloned;
}

bool parseSymbolMap(llvm::StringRef funcKey,
                    const llvm::json::Object &symObj,
                    llvm::StringMap<int64_t> &outMap,
                    std::string &errMsg) {
  for (auto &[symKey, val] : symObj) {
    auto intVal = val.getAsInteger();
    if (!intVal) {
      errMsg = "Value for symbol '" + symKey.str() + "' in '" +
               funcKey.str() + "' must be an integer";
      return false;
    }
    outMap[symKey] = *intVal;
  }
  return true;
}

bool parseBindingList(llvm::StringRef funcKey,
                      const llvm::json::Value &symVals,
                      std::vector<llvm::StringMap<int64_t>> &outBindings,
                      std::string &errMsg) {
  if (auto *symObj = symVals.getAsObject()) {
    llvm::StringMap<int64_t> symMap;
    if (!parseSymbolMap(funcKey, *symObj, symMap, errMsg))
      return false;
    outBindings.push_back(std::move(symMap));
    return true;
  }

  if (auto *bindingArray = symVals.getAsArray()) {
    if (bindingArray->empty()) {
      errMsg = "Value for key '" + funcKey.str() +
               "' must not be an empty array";
      return false;
    }
    for (size_t i = 0; i < bindingArray->size(); ++i) {
      auto *symObj = (*bindingArray)[i].getAsObject();
      if (!symObj) {
        errMsg = "Value for key '" + funcKey.str() + "' at index " +
                 std::to_string(i) + " must be a JSON object";
        return false;
      }
      llvm::StringMap<int64_t> symMap;
      if (!parseSymbolMap(funcKey, *symObj, symMap, errMsg))
        return false;
      outBindings.push_back(std::move(symMap));
    }
    return true;
  }

  errMsg = "Value for key '" + funcKey.str() +
           "' must be a JSON object, array of objects, or null";
  return false;
}

/// Parse the JSON block sizes string into a BlockSizeMap.
/// JSON format: {"func_name": {"SYM": value, ...}, ...}
///          or: {"func_name": [{"SYM": value, ...}, ...], ...}
/// Returns true on success and fills outMap; false on parse error (fills errMsg).
bool parseBlockSizesJson(const char *json_str,
                         loom::passes::BlockSizeMap &outMap,
                         loom::passes::CandidateOrder &candidateOrder,
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
    if (funcKey == "__loom_candidate_order__") {
      auto *order = symVals.getAsArray();
      if (!order) {
        errMsg = "Value for '__loom_candidate_order__' must be an array";
        return false;
      }
      for (const auto &nameValue : *order) {
        auto name = nameValue.getAsString();
        if (!name) {
          errMsg =
              "Entries in '__loom_candidate_order__' must be strings";
          return false;
        }
        candidateOrder.push_back(name->str());
      }
      continue;
    }

    // UNSAT variant: null block sizes → skip (not added to outMap)
    if (symVals.kind() == llvm::json::Value::Null) {
      // llvm::errs() << "info: variant '" << funcKey
      //              << "' is UNSAT, will be omitted from output IR\n";
      continue;
    }
    std::vector<llvm::StringMap<int64_t>> bindings;
    if (!parseBindingList(funcKey, symVals, bindings, errMsg))
      return false;
    outMap[funcKey] = std::move(bindings);
  }
  return true;
}

bool compatibleElementTypes(Type workload, Type hardware) {
  auto workloadShaped = dyn_cast<ShapedType>(workload);
  auto hardwareShaped = dyn_cast<ShapedType>(hardware);
  return workloadShaped && hardwareShaped &&
         workloadShaped.getElementType() == hardwareShaped.getElementType();
}

LogicalResult bindAndValidateCopies(
    ModuleOp module, const loom::lcs::HWOpRegistry &registry,
    std::string &error) {
  bool valid = true;
  module.walk([&](loom::CopyOp copy) {
    if (!valid)
      return;
    auto src = copy.getSrcMemSpaceAttr();
    auto dst = copy.getDstMemSpaceAttr();
    if (!src || !dst || !copy.getArea().empty()) {
      error = "materialized copy has incomplete endpoints or dynamic area";
      valid = false;
      return;
    }

    std::vector<const loom::lcs::HWComputeFunc *> candidates;
    if (copy->hasAttr("loom.unicast_area_unresolved")) {
      candidates = registry.lookupUnicastDataMoverCandidates(
          loom::lcs::DataMoverKind::Copy, src.getLeafReference(),
          dst.getLeafReference());
    } else {
      std::vector<int64_t> area(copy.getStaticArea().begin(),
                                copy.getStaticArea().end());
      candidates = registry.lookupDataMoverCandidates(
          loom::lcs::DataMoverKind::Copy, src.getLeafReference(),
          dst.getLeafReference(), area);
    }
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [&](const auto *candidate) {
                         return !compatibleElementTypes(
                                    copy.getSource().getType(),
                                    candidate->src_type) ||
                                !compatibleElementTypes(
                                    copy.getDestination().getType(),
                                    candidate->dst_type);
                       }),
        candidates.end());
    llvm::sort(candidates, [](const auto *lhs, const auto *rhs) {
      return lhs->registration_order < rhs->registration_order;
    });
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());

    auto array = copy->getAttrOfType<StringAttr>("loom.processor_array");
    auto function =
        copy->getAttrOfType<StringAttr>("loom.processor_function");
    if (static_cast<bool>(array) != static_cast<bool>(function)) {
      error = "materialized copy selection requires both processor array and "
              "function";
      valid = false;
      return;
    }
    const loom::lcs::HWComputeFunc *selected = nullptr;
    if (array) {
      selected = registry.lookupExact(array.getValue(), function.getValue());
      if (!selected || !selected->is_data_mover ||
          !llvm::is_contained(candidates, selected)) {
        error = "materialized copy selection no longer matches its endpoints";
        valid = false;
        return;
      }
    } else if (!candidates.empty()) {
      selected = candidates.back();
    }
    if (!selected) {
      error = "no architect-declared direct mover implements materialized "
              "copy " +
              src.getLeafReference().str() + " -> " +
              dst.getLeafReference().str();
      valid = false;
      return;
    }

    OpBuilder builder(copy);
    if (copy->hasAttr("loom.unicast_area_unresolved")) {
      SmallVector<int64_t> unicastArea(selected->broadcast.size(), 1);
      copy->setAttr("static_area",
                    builder.getDenseI64ArrayAttr(unicastArea));
      copy->removeAttr("loom.unicast_area_unresolved");
    }
    auto setKind = [&](StringRef name, std::optional<int64_t> kind) {
      if (kind)
        copy->setAttr(name, builder.getI64IntegerAttr(*kind));
      else
        copy->removeAttr(name);
    };
    setKind("src_mem_kind", selected->src_mem_kind);
    setKind("dst_mem_kind", selected->dst_mem_kind);
    copy->setAttr("loom.processor_array",
                  builder.getStringAttr(selected->hw_component));
    copy->setAttr("loom.processor_function",
                  builder.getStringAttr(selected->hw_func_name));
  });
  return success(valid);
}

/// Core materialization pipeline logic.
std::pair<std::string, std::string>
runMaterializationCore(const char *input_mlir_text,
                       const char *block_sizes_json,
                       const char *hw_spec_file) {
  // --- Parse block sizes JSON ---
  loom::passes::BlockSizeMap blockSizeMap;
  loom::passes::CandidateOrder candidateOrder;
  std::string errMsg;
  bool hasExternalSizes =
      block_sizes_json && block_sizes_json[0] != '\0';

  if (hasExternalSizes) {
    if (!parseBlockSizesJson(block_sizes_json, blockSizeMap, candidateOrder,
                             errMsg))
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
                      mlir::adl::ADLDialect,
                      func::FuncDialect,
                      arith::ArithDialect,
                      affine::AffineDialect,
                      tensor::TensorDialect,
                      linalg::LinalgDialect,
                      memref::MemRefDialect,
                      scf::SCFDialect,
                      cf::ControlFlowDialect,
                      math::MathDialect,
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

  std::string targetError;
  auto target = loom::lcs::parseTargetEnvironment(targetError);
  if (failed(target))
    return {targetError, ""};
  if (failed(loom::lcs::resolveModuleTarget(*module, *target, targetError)))
    return {targetError, ""};
  const bool bindingsFinalized =
      (*module)->hasAttr("loom.bindings_finalized");

  if (!hw_spec_file || hw_spec_file[0] == '\0')
    return {"materialization requires a hardware specification", ""};
  loom::lcs::HWOpRegistry hardwareRegistry;
  if (failed(hardwareRegistry.loadFromPlatformFile(hw_spec_file, context)))
    return {"Failed to load platform IR from: " + std::string(hw_spec_file),
            ""};

  // Expand special "ALL" binding to every candidate function in the input MLIR.
  // Explicit per-function bindings take precedence over "ALL".
  if (hasExternalSizes) {
    auto allIt = blockSizeMap.find("ALL");
    if (allIt != blockSizeMap.end()) {
      auto allBindings = cloneBindingList(allIt->second);
      // Erase by key before inserting new entries. Inserting into StringMap may
      // rehash and invalidate iterators, so using `allIt` after insertions can
      // trigger LLVM StringMap internal assertions.
      blockSizeMap.erase("ALL");
      candidateOrder.clear();
      module->walk([&](func::FuncOp func) {
        StringRef funcName = func.getName();
        if (blockSizeMap.find(funcName) == blockSizeMap.end()) {
          blockSizeMap[funcName] = cloneBindingList(allBindings);
          candidateOrder.push_back(funcName.str());
        }
      });
    }
  }

  // --- Build pass pipeline ---
  PassManager pm(&context);

  // Stage 1: Materialize symbolic block sizes
  if (hasExternalSizes) {
    pm.addPass(
        loom::passes::createMaterializePass(blockSizeMap, candidateOrder));
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
  if (!bindingsFinalized)
    pm.addPass(loom::passes::createLowerLinalgCopyToLoomCopyPass());

  // Stage 6: Backend-specific TT optimizations
  // (matches tt-opt single-stage driver).
  if (*target == loom::lcs::Target::TT) {
    pm.addPass(loom::passes::createConvertZeroFillLinalgMatmulToLoomPass());
    pm.addPass(loom::passes::createFoldZeroFillLinalgPass());
    pm.addPass(loom::passes::createSplitBinaryScalarChainPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  // --- Run pipeline ---
  if (failed(pm.run(*module)))
    return {"Pipeline execution failed", ""};
  if (failed(bindAndValidateCopies(*module, hardwareRegistry, errMsg)))
    return {errMsg, ""};

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
                           const std::string &block_sizes_json,
                           const std::string &hw_spec_file) {
  return runMaterializationCore(input_mlir_text.c_str(),
                                block_sizes_json.c_str(),
                                hw_spec_file.c_str());
}

} // namespace pipeline
} // namespace loom
