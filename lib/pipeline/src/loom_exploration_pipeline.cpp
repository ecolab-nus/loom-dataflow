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
#include "hw_op_registry.h"
#include "compute_binding.h"
#include "hardware_info.h"
#include "staged_etg_builder.h"
#include "target.h"
#include "driver_utils.h"
#include "lcs_utils.h"
#include "ssa_utils.h"
#include "utils.h"

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
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "ADL/IR/ADLDialect.h"
#include "ADL/IR/ADLTypes.h"
#include "ADL/IR/ADLOps.h"
#include "LoomDialect.h.inc"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

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
    auto isScopeKey = [](llvm::StringRef key) {
      return key == "load_scope" || key == "compute_scope" ||
             key == "store_scope";
    };
    auto writeKV = [&](llvm::StringRef key, const llvm::json::Value &value) {
      os.indent(indent + 2);
      os << "\"" << key << "\": ";
      writeETGJson(os, value, indent + 2);
      if (++i < size)
        os << ",";
      os << "\n";
    };
    static const llvm::StringRef kScopeOrder[] = {
        "load_scope", "compute_scope", "store_scope"};
    for (llvm::StringRef key : kScopeOrder) {
      auto it = obj->find(key);
      if (it != obj->end())
        writeKV(it->first, it->second);
    }
    for (const auto &kv : *obj) {
      if (isScopeKey(kv.first))
        continue;
      writeKV(kv.first, kv.second);
    }
    os.indent(indent) << "}";
    return;
  }

  os << llvm::formatv("{0}", val);
}

/// Build staged ETG JSON from a module and return as a string.
/// Replicates staged_etg_main.cpp logic.
std::pair<std::string, std::string>
buildETGString(ModuleOp module, const loom::lcs::HWOpRegistry &registry,
               loom::lcs::Target target) {
  llvm::json::Array json_etgs;
  bool etgFailed = false;

  module.walk([&](func::FuncOp func_op) {
    if (etgFailed)
      return;
    if (func_op.isExternal() || func_op.empty())
      return;
    loom::lcs::VariantETG etg(func_op.getName(), &registry, target);
    if (failed(etg.buildFromFunc(func_op))) {
      etgFailed = true;
      return;
    }
    if (failed(etg.buildConstraintScope(func_op))) {
      etgFailed = true;
      return;
    }
    json_etgs.push_back(etg.toJSON());
  });

  if (etgFailed)
    return {"ETG construction failed; see MLIR diagnostics above", ""};

  std::string result;
  llvm::raw_string_ostream output(result);
  llvm::json::Value root(std::move(json_etgs));
  writeETGJson(output, root, 0);
  output << "\n";

  return {"", result};
}

struct BindingSite {
  loom::lcs::ComputeBindingSite analysis;
  std::vector<const loom::lcs::HWComputeFunc *> candidates;
};

struct BindingRequirement {
  std::string memory;
  int64_t kind;
};

StringRef allocationMemoryName(StringRef physicalMemory) {
  physicalMemory.consume_front("mem_");
  return physicalMemory;
}

Type withLocalMemoryKind(Type type, int64_t kind) {
  MLIRContext *context = type.getContext();
  if (auto memref = dyn_cast<MemRefType>(type)) {
    Attribute memorySpace =
        kind == 0
            ? Attribute{}
            : IntegerAttr::get(IntegerType::get(context, 64), kind);
    return MemRefType::get(memref.getShape(), memref.getElementType(),
                           memref.getLayout(), memorySpace);
  }
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    SmallVector<NamedAttribute> attributes;
    if (auto dictionary = dyn_cast_or_null<DictionaryAttr>(tensor.getEncoding()))
      for (NamedAttribute attribute : dictionary)
        if (attribute.getName() != "local_mem_kind")
          attributes.push_back(attribute);
    if (kind != 0)
      attributes.push_back(NamedAttribute(
          StringAttr::get(context, "local_mem_kind"),
          IntegerAttr::get(IntegerType::get(context, 64), kind)));
    Attribute encoding =
        attributes.empty() ? Attribute{} : DictionaryAttr::get(context, attributes);
    return RankedTensorType::get(tensor.getShape(), tensor.getElementType(),
                                 encoding);
  }
  return type;
}

bool compatibleShapedTypes(Type workload, Type hardware) {
  auto workloadShaped = dyn_cast<ShapedType>(workload);
  auto hardwareShaped = dyn_cast<ShapedType>(hardware);
  return workloadShaped && hardwareShaped && workloadShaped.hasRank() &&
         hardwareShaped.hasRank() &&
         workloadShaped.getRank() == hardwareShaped.getRank() &&
         workloadShaped.getElementType() == hardwareShaped.getElementType();
}

bool compatibleElementTypes(Type workload, Type hardware) {
  auto workloadShaped = dyn_cast<ShapedType>(workload);
  auto hardwareShaped = dyn_cast<ShapedType>(hardware);
  return workloadShaped && hardwareShaped &&
         workloadShaped.getElementType() == hardwareShaped.getElementType();
}

FailureOr<bool> matchesExplicitResidency(
    const loom::lcs::ComputeBindingSite &site,
    const loom::lcs::HWComputeFunc &candidate) {
  auto linalgOp = cast<linalg::LinalgOp>(site.enclosing_op);
  SmallVector<Value> operands;
  llvm::append_range(operands, linalgOp.getDpsInputs());
  llvm::append_range(operands, linalgOp.getDpsInits());
  auto matchesValue = [&](const loom::lcs::BindingSiteValue &value,
                          unsigned candidateIndex) -> FailureOr<bool> {
    if (value.kind != loom::lcs::BindingSiteValue::ExternalOperand)
      return true;
    if (candidateIndex >= candidate.compute_match.operand_mem_kinds.size() ||
        candidateIndex >= candidate.operand_mem_spaces.size() ||
        value.dps_index >= operands.size())
      return false;
    auto matchesDpsIndex = [&](unsigned dpsIndex) -> FailureOr<bool> {
      if (dpsIndex >= operands.size())
        return false;
      auto explicitKind = loom::lcs::getExplicitLocalMemKind(
          operands[dpsIndex].getType(), site.enclosing_op, dpsIndex);
      if (failed(explicitKind))
        return failure();
      if (*explicitKind &&
          **explicitKind != candidate.compute_match.operand_mem_kinds[candidateIndex])
        return false;
      Value root = loom::utils::traceToRootAlloc(operands[dpsIndex]);
      if (auto alloc = root.getDefiningOp<loom::AllocOp>()) {
        if (auto kind = alloc->getAttrOfType<IntegerAttr>(
                "loom.explicit_local_mem_kind"))
          if (kind.getInt() !=
              candidate.compute_match.operand_mem_kinds[candidateIndex])
            return false;
        if (!alloc->hasAttr("loom.inferred_residency")) {
          std::string memory =
              "mem_" + alloc.getMemory().getLeafReference().str();
          if (memory != candidate.operand_mem_spaces[candidateIndex])
            return false;
        }
      }
      return true;
    };
    auto primary = matchesDpsIndex(value.dps_index);
    if (failed(primary) || !*primary)
      return primary;
    for (unsigned dpsIndex : value.external_dps_indices) {
      auto external = matchesDpsIndex(dpsIndex);
      if (failed(external) || !*external)
        return external;
    }
    return true;
  };

  if (site.is_named) {
    unsigned candidateIndex = 0;
    for (const auto &value : site.inputs) {
      auto matches = matchesValue(value, candidateIndex++);
      if (failed(matches) || !*matches)
        return matches;
    }
    for (const auto &value : site.outputs) {
      auto matches = matchesValue(value, candidateIndex++);
      if (failed(matches) || !*matches)
        return matches;
    }
    return true;
  }
  for (auto [value, candidateIndex] :
       llvm::zip(site.inputs, candidate.scalar_operand_dps_indices)) {
    auto matches = matchesValue(value, candidateIndex);
    if (failed(matches) || !*matches)
      return matches;
  }
  for (auto [value, candidateIndex] :
       llvm::zip(site.outputs, candidate.scalar_result_dps_indices)) {
    auto matches = matchesValue(value, candidateIndex);
    if (failed(matches) || !*matches)
      return matches;
  }
  return true;
}

void collectValues(func::FuncOp function, SmallVectorImpl<Value> &values) {
  for (Block &block : function.getBody())
    values.append(block.args_begin(), block.args_end());
  function.walk([&](Operation *op) {
    values.append(op->result_begin(), op->result_end());
    for (Region &region : op->getRegions())
      for (Block &block : region)
        values.append(block.args_begin(), block.args_end());
  });
}

LogicalResult applyBindingChoice(
    func::FuncOp function, ArrayRef<BindingSite> sites,
    ArrayRef<const loom::lcs::HWComputeFunc *> choice,
    std::string &reason) {
  auto clonedAnalysis = loom::lcs::analyzeComputeBindingSites(function);
  if (failed(clonedAnalysis) || clonedAnalysis->size() != sites.size()) {
    reason = "internal error: cloned binding-site order changed";
    return failure();
  }

  const std::vector<std::string> *processorDomain = nullptr;
  for (const auto *candidate : choice) {
    if (!processorDomain)
      processorDomain = &candidate->processor_domain;
    else if (*processorDomain != candidate->processor_domain) {
      reason = "selected processor families have incompatible spatial domains";
      return failure();
    }
  }

  llvm::DenseMap<Value, BindingRequirement> requirements;
  llvm::DenseMap<Value, BindingRequirement> internalRequirements;
  auto addRequirement = [&](const loom::lcs::BindingSiteValue &siteValue,
                            const BindingRequirement &requirement,
                            Operation *diagnosticOp) -> LogicalResult {
    if (siteValue.kind == loom::lcs::BindingSiteValue::Constant)
      return success();
    if (siteValue.kind == loom::lcs::BindingSiteValue::InternalValue) {
      auto [it, inserted] =
          internalRequirements.try_emplace(siteValue.value, requirement);
      if (!inserted && (it->second.memory != requirement.memory ||
                        it->second.kind != requirement.kind)) {
        reason = "internal producer-consumer edge requires an implicit "
                 "transfer";
        return failure();
      }
      return success();
    }

    auto linalgOp = cast<linalg::LinalgOp>(diagnosticOp);
    SmallVector<Value> dpsOperands;
    llvm::append_range(dpsOperands, linalgOp.getDpsInputs());
    llvm::append_range(dpsOperands, linalgOp.getDpsInits());
    if (siteValue.dps_index >= dpsOperands.size()) {
      reason = "binding site refers to an invalid DPS operand";
      return failure();
    }
    Value operand = dpsOperands[siteValue.dps_index];
    auto explicitKind = loom::lcs::getExplicitLocalMemKind(
        operand.getType(), diagnosticOp, siteValue.dps_index);
    if (failed(explicitKind))
      return failure();
    if (*explicitKind && **explicitKind != requirement.kind) {
      reason = "explicit operand residency conflicts with selected processor";
      return failure();
    }
    Value root = loom::utils::traceToRootAlloc(operand);
    if (!root) {
      reason = "compute operand is not backed by an internal loom.alloc";
      return failure();
    }
    if (auto alloc = root.getDefiningOp<loom::AllocOp>()) {
      if (auto kind = alloc->getAttrOfType<IntegerAttr>(
              "loom.explicit_local_mem_kind")) {
        if (kind.getInt() != requirement.kind) {
          reason = "explicit allocation kind conflicts with selected processor";
          return failure();
        }
      }
      if (!alloc->hasAttr("loom.inferred_residency")) {
        std::string allocated = "mem_" + alloc.getMemory().getLeafReference().str();
        if (allocated != requirement.memory) {
          reason = "explicit allocation memory conflicts with selected processor";
          return failure();
        }
      }
    }
    auto [it, inserted] = requirements.try_emplace(root, requirement);
    if (!inserted && (it->second.memory != requirement.memory ||
                      it->second.kind != requirement.kind)) {
      reason = "shared allocation has conflicting processor residency "
               "requirements";
      return failure();
    }
    return success();
  };

  for (size_t siteIndex = 0; siteIndex < choice.size(); ++siteIndex) {
    const auto &site = (*clonedAnalysis)[siteIndex];
    Operation *op = site.enclosing_op;
    const auto *candidate = choice[siteIndex];
    auto bindValue = [&](const loom::lcs::BindingSiteValue &value,
                         unsigned dpsIndex) -> LogicalResult {
      if (dpsIndex >= candidate->operand_mem_spaces.size() ||
          dpsIndex >= candidate->compute_match.operand_mem_kinds.size()) {
        reason = "selected implementation has incomplete DPS residency metadata";
        return failure();
      }
      BindingRequirement requirement{
          candidate->operand_mem_spaces[dpsIndex],
          candidate->compute_match.operand_mem_kinds[dpsIndex]};
      if (failed(addRequirement(value, requirement, op)))
        return failure();
      for (unsigned externalDpsIndex : value.external_dps_indices) {
        auto external = value;
        external.kind = loom::lcs::BindingSiteValue::ExternalOperand;
        external.dps_index = externalDpsIndex;
        external.external_dps_indices.clear();
        if (failed(addRequirement(external, requirement, op)))
          return failure();
      }
      return success();
    };
    if (site.is_named) {
      auto linalgOp = cast<linalg::LinalgOp>(op);
      SmallVector<Value> operands;
      llvm::append_range(operands, linalgOp.getDpsInputs());
      llvm::append_range(operands, linalgOp.getDpsInits());
      if (candidate->operand_types.size() != operands.size()) {
        reason = "selected named implementation has incompatible DPS arity";
        return failure();
      }
      for (auto [operand, hardwareType] :
           llvm::zip(operands, candidate->operand_types))
        if (!compatibleShapedTypes(operand.getType(), hardwareType)) {
          reason = "selected named implementation has incompatible operand "
                   "rank or element type";
          return failure();
        }
      unsigned dpsIndex = 0;
      for (const auto &input : site.inputs)
        if (failed(bindValue(input, dpsIndex++)))
          return failure();
      for (const auto &output : site.outputs)
        if (failed(bindValue(output, dpsIndex++)))
          return failure();
    } else {
      if (candidate->scalar_operand_dps_indices.size() != site.inputs.size() ||
          candidate->scalar_result_dps_indices.size() != site.outputs.size()) {
        reason = "selected primitive has incompatible scalar/DPS correspondence";
        return failure();
      }
      for (auto [input, dpsIndex] :
           llvm::zip(site.inputs, candidate->scalar_operand_dps_indices))
        if (failed(bindValue(input, dpsIndex)))
          return failure();
      for (auto [output, dpsIndex] :
           llvm::zip(site.outputs, candidate->scalar_result_dps_indices))
        if (failed(bindValue(output, dpsIndex)))
          return failure();
    }
  }

  // Helion's stage-02 lowering may hand a loop-carried result to a distinct
  // write-back buffer with linalg.copy. This is a same-residency storage
  // handoff, so propagate the selected physical memory across it. A conflict
  // still requires storage splitting and is deliberately unsupported.
  bool changed = true;
  while (changed) {
    changed = false;
    function.walk([&](linalg::CopyOp copy) {
      auto linalgOp = cast<linalg::LinalgOp>(copy.getOperation());
      if (linalgOp.getDpsInputs().size() != 1 ||
          linalgOp.getDpsInits().size() != 1)
        return;
      Value sourceRoot =
          loom::utils::traceToRootAlloc(linalgOp.getDpsInputs().front());
      Value destinationRoot =
          loom::utils::traceToRootAlloc(linalgOp.getDpsInits().front());
      if (!sourceRoot || !destinationRoot)
        return;
      auto source = requirements.find(sourceRoot);
      auto destination = requirements.find(destinationRoot);
      if (source != requirements.end() && destination == requirements.end()) {
        requirements.try_emplace(destinationRoot, source->second);
        changed = true;
      } else if (source == requirements.end() &&
                 destination != requirements.end()) {
        requirements.try_emplace(sourceRoot, destination->second);
        changed = true;
      }
    });
  }
  bool handoffConflict = false;
  function.walk([&](linalg::CopyOp copy) {
    auto linalgOp = cast<linalg::LinalgOp>(copy.getOperation());
    if (linalgOp.getDpsInputs().size() != 1 ||
        linalgOp.getDpsInits().size() != 1)
      return;
    Value sourceRoot =
        loom::utils::traceToRootAlloc(linalgOp.getDpsInputs().front());
    Value destinationRoot =
        loom::utils::traceToRootAlloc(linalgOp.getDpsInits().front());
    auto source = requirements.find(sourceRoot);
    auto destination = requirements.find(destinationRoot);
    if (source != requirements.end() && destination != requirements.end() &&
        (source->second.memory != destination->second.memory ||
         source->second.kind != destination->second.kind))
      handoffConflict = true;
  });
  if (handoffConflict) {
    reason = "linalg.copy handoff has conflicting processor residency "
             "requirements";
    return failure();
  }

  SmallVector<Value> values;
  collectValues(function, values);
  for (const auto &[root, requirement] : requirements) {
    auto alloc = root.getDefiningOp<loom::AllocOp>();
    if (!alloc) {
      reason = "binding root is not loom.alloc";
      return failure();
    }
    SmallVector<Value> aliases;
    for (Value value : values)
      if (isa<ShapedType>(value.getType()) &&
          loom::utils::traceToRootAlloc(value) == root)
        aliases.push_back(value);
    alloc->setAttr("memory",
                   FlatSymbolRefAttr::get(function.getContext(),
                                          allocationMemoryName(
                                              requirement.memory)));
    for (Value alias : aliases)
      alias.setType(withLocalMemoryKind(alias.getType(), requirement.kind));
  }

  for (size_t siteIndex = 0; siteIndex < choice.size(); ++siteIndex) {
    Operation *op = (*clonedAnalysis)[siteIndex].payload_op;
    const auto *candidate = choice[siteIndex];
    op->setAttr("loom.processor_array",
                StringAttr::get(function.getContext(),
                                candidate->hw_component));
    op->setAttr("loom.processor_function",
                StringAttr::get(function.getContext(),
                                candidate->hw_func_name));
    op->setAttr("loom.binding_site",
                IntegerAttr::get(IntegerType::get(function.getContext(), 64),
                                 sites[siteIndex].analysis.id));
  }

  function.walk([&](loom::CopyOp copy) {
    Value sourceRoot = loom::utils::traceToRootAlloc(copy.getSource());
    Value destinationRoot =
        loom::utils::traceToRootAlloc(copy.getDestination());
    auto updateEndpoint = [&](Value root, StringRef spaceAttr,
                              StringRef kindAttr) {
      auto it = requirements.find(root);
      if (it == requirements.end())
        return;
      copy->setAttr(spaceAttr, FlatSymbolRefAttr::get(
                                   function.getContext(), it->second.memory));
      // Data-mover-local kind attributes are part of the mover interface, not
      // the compute binding. They are filled after a mover is selected.
      copy->removeAttr(kindAttr);
    };
    updateEndpoint(sourceRoot, "src_mem_space", "src_mem_kind");
    updateEndpoint(destinationRoot, "dst_mem_space", "dst_mem_kind");
    copy->removeAttr("loom.processor_array");
    copy->removeAttr("loom.processor_function");
  });
  return success();
}

LogicalResult assignMoverChoice(
    func::FuncOp function,
    ArrayRef<const loom::lcs::HWComputeFunc *> choice,
    std::string &reason) {
  SmallVector<loom::CopyOp> copies;
  function.walk([&](loom::CopyOp copy) { copies.push_back(copy); });
  if (copies.size() != choice.size()) {
    reason = "internal error: cloned copy-operation order changed";
    return failure();
  }
  llvm::json::Array manifest;
  for (auto [copy, mover] : llvm::zip(copies, choice)) {
    auto setKind = [&](StringRef name, std::optional<int64_t> kind) {
      if (kind)
        copy->setAttr(name, IntegerAttr::get(
                                IntegerType::get(function.getContext(), 64),
                                *kind));
      else
        copy->removeAttr(name);
    };
    setKind("src_mem_kind", mover->src_mem_kind);
    setKind("dst_mem_kind", mover->dst_mem_kind);
    copy->setAttr("loom.processor_array",
                  StringAttr::get(function.getContext(), mover->hw_component));
    copy->setAttr("loom.processor_function",
                  StringAttr::get(function.getContext(), mover->hw_func_name));
    manifest.push_back(llvm::json::Object{
        {"array", mover->hw_component},
        {"function", mover->hw_func_name},
        {"source", mover->src_mem_space},
        {"destination", mover->dst_mem_space}});
  }
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << llvm::json::Value(std::move(manifest));
  function->setAttr("loom.mover_manifest",
                    StringAttr::get(function.getContext(), stream.str()));
  return success();
}

LogicalResult finalizeDataMoverVariants(
    ModuleOp module, const loom::lcs::HWOpRegistry &registry,
    bool enumerate, std::string &error) {
  SmallVector<func::FuncOp> originals =
      loom::utils::collectFunctions(module);
  OpBuilder builder(module.getBodyRegion());
  llvm::json::Array rejections;
  size_t legalCount = 0;

  for (func::FuncOp original : originals) {
    SmallVector<loom::CopyOp> copies;
    original.walk([&](loom::CopyOp copy) { copies.push_back(copy); });
    if (copies.empty()) {
      ++legalCount;
      continue;
    }

    std::vector<std::vector<const loom::lcs::HWComputeFunc *>> alternatives;
    bool valid = true;
    std::string rejection;
    for (loom::CopyOp copy : copies) {
      auto srcSpace = copy.getSrcMemSpaceAttr();
      auto dstSpace = copy.getDstMemSpaceAttr();
      if (!srcSpace || !dstSpace || !copy.getArea().empty()) {
        rejection = "transfer has incomplete endpoints or dynamic area";
        valid = false;
        break;
      }
      std::vector<int64_t> area(copy.getStaticArea().begin(),
                                copy.getStaticArea().end());
      auto candidates = registry.lookupDataMoverCandidates(
          loom::lcs::DataMoverKind::Copy, srcSpace.getLeafReference(),
          dstSpace.getLeafReference(), area);
      candidates.erase(
          std::remove_if(candidates.begin(), candidates.end(),
                         [&](const auto *mover) {
                           return !compatibleElementTypes(
                                      copy.getSource().getType(),
                                      mover->src_type) ||
                                  !compatibleElementTypes(
                                      copy.getDestination().getType(),
                                      mover->dst_type);
                         }),
          candidates.end());
      llvm::sort(candidates, [](const auto *lhs, const auto *rhs) {
        return lhs->registration_order < rhs->registration_order;
      });
      candidates.erase(std::unique(candidates.begin(), candidates.end()),
                       candidates.end());
      if (candidates.empty()) {
        rejection = "no architect-declared direct mover implements transfer " +
                    srcSpace.getLeafReference().str() + " -> " +
                    dstSpace.getLeafReference().str();
        valid = false;
        break;
      }
      if (!enumerate)
        candidates.assign(1, candidates.back());
      alternatives.push_back(std::move(candidates));
    }

    if (!valid) {
      if (!enumerate) {
        error = "fixed data-mover selection failed for variant '" +
                original.getName().str() + "': " + rejection;
        return failure();
      }
      rejections.push_back(llvm::json::Object{
          {"variant", original.getName().str()}, {"rejected", rejection}});
      if (ModuleOp parent = loom::utils::getParentModule(original))
        parent.erase();
      else
        original.erase();
      continue;
    }

    if (!enumerate) {
      std::vector<const loom::lcs::HWComputeFunc *> choice;
      choice.reserve(alternatives.size());
      for (const auto &candidates : alternatives)
        choice.push_back(candidates.back());
      std::string assignmentError;
      if (failed(assignMoverChoice(original, choice, assignmentError))) {
        error = "fixed data-mover selection failed for variant '" +
                original.getName().str() + "': " + assignmentError;
        return failure();
      }
      ++legalCount;
      continue;
    }

    std::vector<std::vector<const loom::lcs::HWComputeFunc *>> choices(1);
    for (const auto &candidates : alternatives) {
      std::vector<std::vector<const loom::lcs::HWComputeFunc *>> next;
      for (const auto &partial : choices)
        for (const auto *candidate : candidates) {
          next.push_back(partial);
          next.back().push_back(candidate);
        }
      choices = std::move(next);
    }

    ModuleOp parent = loom::utils::getParentModule(original);
    DictionaryAttr moduleAttrs = parent ? parent->getAttrDictionary() : nullptr;
    Operation *insertAfter = parent ? static_cast<Operation *>(parent)
                                    : static_cast<Operation *>(original);
    for (size_t choiceIndex = 0; choiceIndex < choices.size(); ++choiceIndex) {
      std::string name = (original.getName() + "__movers_" +
                          Twine(choiceIndex)).str();
      std::string assignmentError;
      func::FuncOp clone = loom::utils::cloneFunc(
          builder, original, name, moduleAttrs,
          [&](func::FuncOp function) {
            return assignMoverChoice(function, choices[choiceIndex],
                                     assignmentError);
          },
          insertAfter);
      if (!clone) {
        rejections.push_back(llvm::json::Object{
            {"variant", name}, {"rejected", assignmentError}});
        continue;
      }
      insertAfter = loom::utils::getParentModule(clone);
      ++legalCount;
    }
    if (parent)
      parent.erase();
    else
      original.erase();
  }

  std::string rejectionText;
  if (!rejections.empty()) {
    llvm::raw_string_ostream stream(rejectionText);
    stream << llvm::json::Value(std::move(rejections));
    module->setAttr("loom.binding_rejections",
                    StringAttr::get(module.getContext(), stream.str()));
  }
  if (legalCount == 0) {
    error = "all variants were rejected during direct data-mover selection: " +
            rejectionText;
    return failure();
  }
  return success();
}

LogicalResult enumerateBindingVariants(
    ModuleOp module, const loom::lcs::HWOpRegistry &registry, bool enumerate,
    std::string &error) {
  SmallVector<func::FuncOp> originals;
  module.walk([&](func::FuncOp function) {
    if (function->getParentOp() == module && !function.isExternal())
      originals.push_back(function);
  });

  llvm::json::Array manifest;
  size_t legalCount = 0;
  for (func::FuncOp original : originals) {
    std::vector<BindingSite> sites;
    auto analysis = loom::lcs::analyzeComputeBindingSites(original);
    if (failed(analysis)) {
      error = "failed to analyze processor binding sites for function '" +
              original.getName().str() + "'";
      return failure();
    }
    bool collectionFailed = false;
    for (auto &site : *analysis) {
      auto candidates = registry.lookupComputeCandidates(site.semantic_key);
      candidates.erase(
          std::remove_if(candidates.begin(), candidates.end(),
                         [&](const auto *candidate) {
                           return !loom::lcs::candidateMatchesSite(*candidate,
                                                                  site);
                         }),
          candidates.end());
      auto selectedArray = site.payload_op->getAttrOfType<StringAttr>(
          "loom.processor_array");
      auto selectedFunction = site.payload_op->getAttrOfType<StringAttr>(
          "loom.processor_function");
      if (selectedArray || selectedFunction) {
        if (!selectedArray || !selectedFunction) {
          site.payload_op->emitError()
              << "processor selection requires both loom.processor_array and "
                 "loom.processor_function";
          collectionFailed = true;
          break;
        }
        const auto *selected = registry.lookupExact(selectedArray.getValue(),
                                                     selectedFunction.getValue());
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                           [&](const auto *candidate) {
                             return candidate != selected;
                           }),
            candidates.end());
      }
      if (!enumerate) {
        bool residencyReadFailed = false;
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                           [&](const auto *candidate) {
                             auto matches =
                                 matchesExplicitResidency(site, *candidate);
                             if (failed(matches)) {
                               residencyReadFailed = true;
                               return true;
                             }
                             return !*matches;
                           }),
            candidates.end());
        if (residencyReadFailed) {
          collectionFailed = true;
          break;
        }
      }
      if (candidates.empty()) {
        site.payload_op->emitError()
            << "processor binding found no semantically compatible MLAR "
               "primitive implementation";
        collectionFailed = true;
        break;
      }
      llvm::sort(candidates, [](const auto *lhs, const auto *rhs) {
        return lhs->registration_order < rhs->registration_order;
      });
      candidates.erase(std::unique(candidates.begin(), candidates.end()),
                       candidates.end());
      site.payload_op->setAttr(
          "loom.binding_site",
          IntegerAttr::get(IntegerType::get(site.payload_op->getContext(), 64),
                           site.id));
      sites.push_back({std::move(site), std::move(candidates)});
    }
    if (collectionFailed) {
      error = "failed to resolve processor bindings for function '" +
              original.getName().str() + "'";
      return failure();
    }
    if (sites.empty()) {
      error = "binding input has no supported compute operations";
      return failure();
    }

    if (!enumerate) {
      std::vector<const loom::lcs::HWComputeFunc *> choice;
      choice.reserve(sites.size());
      for (const BindingSite &site : sites)
        choice.push_back(site.candidates.back());
      std::string rejection;
      if (failed(applyBindingChoice(original, sites, choice, rejection)) ||
          failed(verify(original))) {
        if (rejection.empty())
          rejection = "rebound MLIR verification failed";
        error = "fixed processor binding failed for function '" +
                original.getName().str() + "': " + rejection;
        return failure();
      }
      ++legalCount;
      continue;
    }

    size_t combinationCount = 1;
    for (const BindingSite &site : sites)
      combinationCount *= site.candidates.size();

    std::vector<std::vector<const loom::lcs::HWComputeFunc *>> choices;
    std::vector<const loom::lcs::HWComputeFunc *> partial;
    OpBuilder builder(original);
    std::function<void(size_t)> explore = [&](size_t depth) {
      if (depth == sites.size()) {
        choices.push_back(partial);
        return;
      }
      for (const auto *candidate : sites[depth].candidates) {
        partial.push_back(candidate);
        IRMapping mapping;
        auto probe = cast<func::FuncOp>(builder.clone(*original, mapping));
        std::string rejection;
        if (failed(applyBindingChoice(probe, sites, partial, rejection))) {
          manifest.push_back(llvm::json::Object{
              {"site", static_cast<int64_t>(depth)},
              {"rejected", rejection}});
        } else {
          explore(depth + 1);
        }
        probe.erase();
        partial.pop_back();
      }
    };
    explore(0);

    llvm::json::Array siteCounts;
    for (const BindingSite &site : sites)
      siteCounts.push_back(llvm::json::Object{
          {"site", static_cast<int64_t>(site.analysis.id)},
          {"operation",
           site.analysis.payload_op->getName().getStringRef().str()},
          {"candidate_count",
           static_cast<int64_t>(site.candidates.size())}});
    manifest.push_back(llvm::json::Object{
        {"function", original.getName().str()},
        {"site_candidates", std::move(siteCounts)},
        {"binding_combination_count", static_cast<int64_t>(combinationCount)}});

    const size_t legalBeforeFunction = legalCount;
    for (size_t choiceIndex = 0; choiceIndex < choices.size(); ++choiceIndex) {
      IRMapping mapping;
      auto clone = cast<func::FuncOp>(builder.clone(*original, mapping));
      clone.setName((original.getName() + "__binding_" +
                     Twine(choiceIndex)).str());
      clone->setAttr("loom.binding_preference",
                     builder.getI64IntegerAttr(choiceIndex));
      clone->setAttr("loom.binding_family",
                     builder.getStringAttr(original.getName()));
      std::string rejection;
      if (failed(applyBindingChoice(clone, sites, choices[choiceIndex],
                                    rejection)) ||
          failed(verify(clone))) {
        if (rejection.empty())
          rejection = "rebound MLIR verification failed";
        manifest.push_back(llvm::json::Object{
            {"candidate", static_cast<int64_t>(choiceIndex)},
            {"variant", clone.getName().str()}, {"rejected", rejection}});
        clone.erase();
        continue;
      }
      llvm::json::Array selections;
      for (size_t siteIndex = 0; siteIndex < sites.size(); ++siteIndex) {
        const auto *candidate = choices[choiceIndex][siteIndex];
        llvm::json::Array memories;
        llvm::json::Array kinds;
        for (const std::string &memory : candidate->operand_mem_spaces)
          memories.push_back(memory);
        for (int64_t kind : candidate->compute_match.operand_mem_kinds)
          kinds.push_back(kind);
        selections.push_back(llvm::json::Object{
            {"site", static_cast<int64_t>(sites[siteIndex].analysis.id)},
            {"definition", candidate->processor_definition},
            {"function", candidate->hw_func_name},
            {"array", candidate->hw_component},
            {"operand_memories", std::move(memories)},
            {"operand_kinds", std::move(kinds)}});
      }
      manifest.push_back(llvm::json::Object{
          {"candidate", static_cast<int64_t>(choiceIndex)},
          {"variant", clone.getName().str()},
          {"selections", std::move(selections)}});
      ++legalCount;
    }
    if (legalCount == legalBeforeFunction) {
      std::string manifestText;
      llvm::raw_string_ostream manifestStream(manifestText);
      manifestStream << llvm::json::Value(std::move(manifest));
      error = "all processor-binding candidates for function '" +
              original.getName().str() + "' were rejected: " +
              manifestStream.str();
      return failure();
    }
    original.erase();
  }
  std::string manifestText;
  llvm::raw_string_ostream manifestStream(manifestText);
  manifestStream << llvm::json::Value(std::move(manifest));
  if (legalCount == 0) {
    error = "all processor-binding candidates were rejected: " +
            manifestStream.str();
    return failure();
  }
  if (enumerate)
    module->setAttr("loom.binding_manifest",
                    StringAttr::get(module.getContext(), manifestStream.str()));
  return success();
}

} // namespace

namespace loom {
namespace pipeline {

std::tuple<std::string, std::string, std::string>
runExplorationPipeline(const std::string &input_mlir_text,
                       const std::string &hw_spec_file,
                       bool produce_etg,
                       bool skip_etg,
                       bool full_occ,
                       bool spatial_reuse,
                       bool explicit_memory,
                       bool enumerate_bindings) {
  std::string targetError;
  auto target = loom::lcs::parseTargetEnvironment(targetError);
  if (failed(target))
    return {targetError, "", ""};
  // --- Set up MLIRContext with all required dialects ---
  DialectRegistry registry;
  registry.insert<BuiltinDialect, func::FuncDialect, affine::AffineDialect,
                  memref::MemRefDialect, arith::ArithDialect,
                  tensor::TensorDialect, linalg::LinalgDialect,
                  scf::SCFDialect, bufferization::BufferizationDialect,
                  cf::ControlFlowDialect, math::MathDialect,
                  mlir::adl::ADLDialect, loom::LoomDialect>();

  // Explicitly register missing tensor op external models
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // --- Parse input MLIR from string ---
  auto inputBuf = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(input_mlir_text), "input_mlir");

  llvm::SourceMgr inputSm;
  inputSm.AddNewSourceBuffer(std::move(inputBuf), llvm::SMLoc());
  auto inputModule = parseSourceFile<ModuleOp>(inputSm, &context);
  if (!inputModule)
    return {"Failed to parse input MLIR text", "", ""};
  if (failed(loom::lcs::resolveModuleTarget(*inputModule, *target,
                                            targetError)))
    return {targetError, "", ""};

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

  loom::lcs::HWOpRegistry computeRegistry;
  if (mlir::failed(computeRegistry.loadFromPlatformFile(hw_spec_file, context)))
    return {"Failed to load platform IR from: " + hw_spec_file, "", ""};

  // ================================================================
  // Phase A1: tensor_canonicalize (stage 0→1)
  // ================================================================
  if (!explicit_memory) {
    PassManager pm(&context);

    // Match tensor_canonicalize_main.cpp.
    pm.addPass(loom::passes::createLinalgGuardedElementwiseOpFusionPass());
    pm.addPass(loom::passes::createLinalgDestinationSpecializationPass());
    pm.addPass(loom::passes::createFoldRedundantExtractSlicePass());
    pm.addPass(createSymbolDCEPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(loom::passes::createSinkPreparationOpsPass());
    pm.addPass(loom::passes::createLoopHandoffProxyCopyInsertionPass());
    pm.addPass(loom::passes::createCanonicalBufferizationToLoomPass());

    if (failed(pm.run(*inputModule)))
      return {"Phase A1 (tensor_canonicalize) failed", "", ""};
  }

  // Strip cf.assert guards so memory_binding can run cf-free, mirroring
  // tensor_canonicalize_main.cpp output contract.
  if (!explicit_memory) {
    SmallVector<Operation *> assertsToErase;
    inputModule->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "cf.assert")
        assertsToErase.push_back(op);
    });
    for (Operation *op : assertsToErase)
      op->erase();
  }

  // ================================================================
  // Phase A2: memory_binding (stage 1→2)
  // ================================================================
  if (!explicit_memory) {
    PassManager pm(&context);
    pm.addPass(loom::passes::createMemoryBindingPass());
    if (failed(pm.run(*inputModule)))
      return {"Phase A2 (memory_binding) failed", "", ""};
  }

  {
    std::string bindingError;
    if (failed(enumerateBindingVariants(*inputModule, computeRegistry,
                                        enumerate_bindings, bindingError)))
      return {"Processor binding failed: " + bindingError, "", ""};
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
      loom::EnumerateSpatialMappings(*inputModule, hardwareInfo, full_occ);

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

  // Clean up merged output, matching enumerate_hw_mapping_main.cpp.
  {
    PassManager pm(&context);
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(pm.run(*merged)))
      return {"enumerate_hw_mapping cleanup failed", "", ""};
  }

  // Release intermediate modules to free memory.
  inputModule = nullptr;
  enumerated = nullptr;

  if (spatial_reuse) {
    // ================================================================
    // Phase B: analyze_reuse + enumerate_copy_broadcast (stages 3→5)
    // ================================================================
    PassManager pm(&context);
    pm.addPass(loom::passes::createAnnotateSubviewReusePass());
    pm.addPass(loom::passes::createEnumerateCopyBroadcastPass());
    // Match enumerate_copy_broadcast_main.cpp post-pass cleanup.
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());

    if (failed(pm.run(*merged)))
      return {"Phase B (analyze_reuse + enumerate_copy_broadcast) failed",
              "", ""};
  }

  {
    std::string moverError;
    if (failed(finalizeDataMoverVariants(*merged, computeRegistry,
                                         enumerate_bindings, moverError)))
      return {"Data-mover binding failed: " + moverError, "", ""};
  }
  (*merged)->setAttr("loom.bindings_finalized", UnitAttr::get(&context));

  // ================================================================
  // Optional: Build staged ETG JSON
  // ================================================================
  std::string etg_json;
  const bool shouldProduceEtg = produce_etg && !skip_etg;
  if (shouldProduceEtg) {
    auto [etgErr, etgText] = buildETGString(*merged, computeRegistry, *target);
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
