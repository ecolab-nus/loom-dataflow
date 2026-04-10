/// Implementation of the Loom exploration pipeline.
///
/// Consolidates the following single-stage drivers into one in-memory pipeline:
///   1. tensor_canonicalize  (stages 0→1)
///   2. memory_binding       (stages 1→2)
///   3. enumerate_hw_mapping (stages 2→3)  -- custom logic, not a simple pass
///   4. analyze_reuse        (stages 3→4)
///   5. enumerate_copy_broadcast (stages 4→5)
///   6. (optional) staged_etg   -- ETG JSON extraction

#include "loom_exploration_pipeline.h"
#include "Passes.h"
#include "compute_op_registry.h"
#include "hardware_info.h"
#include "staged_etg_builder.h"
#include "driver_utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "ADLDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"
#include "LoomDialect.h.inc"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <string>
#include <tuple>
#include <utility>

using namespace mlir;

namespace {

/// Custom pretty-printer for ETG JSON.
/// Replicates the formatting from staged_etg_main.cpp.
void writeETGJson(llvm::raw_ostream &os, const llvm::json::Value &val,
                  int indent) {
  auto isExprKind = [](llvm::StringRef key) -> bool {
    static const llvm::StringRef kKinds[] = {
        "Const", "Sym",    "Add",       "Sub",  "Mul",    "Div",
        "Min",   "Max",    "IfElse",    "And",  "Or",     "Not",
        "Eq",    "Le",     "Lt",        "Ge",   "Gt",     "Divisible",
        "InRange",
    };
    for (auto k : kKinds)
      if (k == key)
        return true;
    return false;
  };

  auto isExprNode = [&](const llvm::json::Value &v) -> bool {
    if (const auto *obj = v.getAsObject())
      if (obj->size() == 1)
        return isExprKind(obj->begin()->first);
    return false;
  };

  if (isExprNode(val)) {
    os << llvm::formatv("{0}", val);
    return;
  }

  if (const auto *arr = val.getAsArray()) {
    if (arr->empty()) {
      os << "[]";
      return;
    }
    os << "[\n";
    for (size_t i = 0; i < arr->size(); ++i) {
      os.indent(indent + 2);
      writeETGJson(os, (*arr)[i], indent + 2);
      if (i + 1 < arr->size())
        os << ",";
      os << "\n";
    }
    os.indent(indent) << "]";
    return;
  }

  if (const auto *obj = val.getAsObject()) {
    if (obj->empty()) {
      os << "{}";
      return;
    }
    os << "{\n";
    size_t i = 0, size = obj->size();
    for (const auto &kv : *obj) {
      os.indent(indent + 2);
      llvm::StringRef key = kv.first;
      os << "\"" << key << "\": ";
      writeETGJson(os, kv.second, indent + 2);
      if (++i < size)
        os << ",";
      os << "\n";
    }
    os.indent(indent) << "}";
    return;
  }

  os << llvm::formatv("{0}", val);
}

/// Build staged ETG JSON from a module and return as a string.
/// Replicates staged_etg_main.cpp logic.
std::pair<std::string, std::string>
buildETGString(ModuleOp module, const loom::lcs::HWOpRegistry &registry) {
  llvm::json::Array json_etgs;

  module.walk([&](func::FuncOp func_op) {
    scf::ForOp target_loop = nullptr;

    func_op.walk([&](scf::ForOp for_op) {
      if (for_op->hasAttr("loom.iter_type")) {
        std::string attr_str;
        llvm::raw_string_ostream os(attr_str);
        for_op->getAttr("loom.iter_type").print(os);
        if (attr_str.find("sequential") != std::string::npos) {
          target_loop = for_op;
        }
      }
    });

    if (target_loop) {
      loom::lcs::VariantETG etg(func_op.getName(), &registry);
      etg.buildFromSCFFor(target_loop);
      etg.buildConstraintScope(func_op);
      etg.buildL1FootprintConstraint();
      json_etgs.push_back(etg.toJSON());
    }
  });

  std::string result;
  llvm::raw_string_ostream output(result);
  llvm::json::Value root(std::move(json_etgs));
  writeETGJson(output, root, 0);
  output << "\n";

  return {"", result};
}

} // namespace

namespace loom {
namespace pipeline {

std::tuple<std::string, std::string, std::string>
runExplorationPipeline(const std::string &input_mlir_text,
                       const std::string &hw_spec_file,
                       bool produce_etg) {
  // --- Set up MLIRContext with all required dialects ---
  DialectRegistry registry;
  registry.insert<BuiltinDialect, func::FuncDialect, affine::AffineDialect,
                  memref::MemRefDialect, arith::ArithDialect,
                  tensor::TensorDialect, linalg::LinalgDialect,
                  scf::SCFDialect, bufferization::BufferizationDialect,
                  cf::ControlFlowDialect, math::MathDialect,
                  adl::ADLDialect, loom::LoomDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // --- Load hardware compute IR registry ---
  loom::lcs::HWOpRegistry computeRegistry;
  if (mlir::failed(computeRegistry.loadFromPlatformFile(hw_spec_file, context)))
    return {"Failed to load platform IR from: " + hw_spec_file, "", ""};

  // --- Parse input MLIR from string ---
  auto inputBuf = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(input_mlir_text), "input_mlir");

  llvm::SourceMgr inputSm;
  inputSm.AddNewSourceBuffer(std::move(inputBuf), llvm::SMLoc());
  auto inputModule = parseSourceFile<ModuleOp>(inputSm, &context);
  if (!inputModule)
    return {"Failed to parse input MLIR text", "", ""};

  // --- Parse DF module (hardware description) ---
  auto dfBuf = llvm::MemoryBuffer::getFile(hw_spec_file);
  if (std::error_code ec = dfBuf.getError())
    return {"Could not open DF file '" + hw_spec_file + "': " + ec.message(),
            "", ""};

  llvm::SourceMgr dfSm;
  dfSm.AddNewSourceBuffer(std::move(*dfBuf), llvm::SMLoc());
  auto dfModule = parseSourceFile<ModuleOp>(dfSm, &context);
  if (!dfModule)
    return {"Failed to parse DF MLIR file: " + hw_spec_file, "", ""};

  // ================================================================
  // Phase A: tensor_canonicalize + memory_binding (stages 0→2)
  // ================================================================
  // The input module is wrapped in an outer module { module { func ... } }.
  // The passes operate on the inner module's func ops.
  {
    PassManager pm(&context);

    // -- tensor_canonicalize passes --
    pm.addPass(createLinalgElementwiseOpFusionPass());
    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(loom::passes::createLinalgDestinationSpecializationPass());
    pm.addPass(createSymbolDCEPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(loom::passes::createFoldRedundantExtractSlicePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(loom::passes::createSinkFillOpsPass());

    // -- memory_binding pass --
    pm.addPass(loom::passes::createMemoryBindingPass());

    if (failed(pm.run(*inputModule)))
      return {"Phase A (tensor_canonicalize + memory_binding) failed", "", ""};
  }

  // ================================================================
  // Interlude: enumerate_hw_mapping (stage 2→3)
  // This is NOT a standard pass — it creates a new ModuleOp.
  // Replicates enumerate_hw_mapping_main.cpp lines 92-141.
  // ================================================================

  // Collect hardware info from DF module.
  loom::HardwareInfo hardwareInfo;
  if (failed(loom::GetHardwareInfoForExploration(*dfModule, hardwareInfo)))
    return {"Failed to collect hardware information from DF module", "", ""};

  // Enumerate spatial mappings — returns a brand new ModuleOp.
  OwningOpRef<ModuleOp> enumerated =
      loom::EnumerateSpatialMappings(*inputModule, hardwareInfo);

  // Merge DF declarations and enumerated clones into a single module.
  OwningOpRef<ModuleOp> merged =
      ModuleOp::create(UnknownLoc::get(&context));
  if (!(*enumerated)->getAttrs().empty())
    (*merged)->setAttrs((*enumerated)->getAttrs());

  {
    OpBuilder builder(merged->getBodyRegion());
    IRMapping mapping;

    // Insert DF hardware declarations from the hardware specification at the
    // top of the outer module.
    // We use findArchSystemModule to locate the @arch_system module.
    ModuleOp systemModule = loom::driver::findArchSystemModule(*dfModule);
    if (!systemModule)
      return {"Could not find module @arch_system in hw_spec file", "", ""};

    for (Operation &op : *systemModule.getBody()) {
      if (isa<ModuleOp>(&op))
          continue;
      if (op.hasTrait<OpTrait::IsTerminator>())
          continue;
      builder.clone(op, mapping);
    }

    // Insert all the nested modules containing function variants.
    for (Operation &op : *enumerated->getBody())
      builder.clone(op, mapping);
  }

  // Release intermediate modules to free memory.
  inputModule = nullptr;
  enumerated = nullptr;

  // ================================================================
  // Phase B: analyze_reuse + enumerate_copy_broadcast (stages 3→5)
  // ================================================================
  {
    PassManager pm(&context);
    // Cleanup the generated code with CSE and DCE (Canonicalizer)
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());

    pm.addPass(loom::passes::createAnnotateSubviewReusePass());
    pm.addPass(loom::passes::createEnumerateCopyBroadcastPass());

    if (failed(pm.run(*merged)))
      return {"Phase B (analyze_reuse + enumerate_copy_broadcast) failed",
              "", ""};
  }

  // ================================================================
  // Optional: Build staged ETG JSON
  // ================================================================
  std::string etg_json;
  if (produce_etg) {
    auto [etgErr, etgText] = buildETGString(*merged, computeRegistry);
    if (!etgErr.empty())
      return {etgErr, "", ""};
    etg_json = std::move(etgText);
  }

  // ================================================================
  // Serialize output MLIR to string
  // ================================================================
  std::string output_mlir;
  llvm::raw_string_ostream outStream(output_mlir);

  OpPrintingFlags flags;
  flags.useLocalScope();
  merged->print(outStream, flags);
  outStream << "\n";

  return {"", output_mlir, etg_json};
}

} // namespace pipeline
} // namespace loom
