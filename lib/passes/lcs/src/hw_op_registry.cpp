/**
 * @file hw_op_registry.cpp
 * @brief Implementation of HWOpRegistry for loading and indexing
 *        hardware platform IR files.
 */

#include "hw_op_registry.h"
#include "hw_op_registry_detail.h"
#include "utils.h"
#include "ADL/IR/ADLDialect.h"
#include "ADL/IR/ADLTypes.h"
#include "ADL/IR/ADLOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

namespace loom {
namespace lcs {

// ============================================================
// HWOpKey
// ============================================================

bool HWOpKey::operator<(const HWOpKey &rhs) const {
  if (kind != rhs.kind)
    return kind < rhs.kind;
  switch (kind) {
  case Named:
    if (linalg_op_name != rhs.linalg_op_name)
      return linalg_op_name < rhs.linalg_op_name;
    return compute_match < rhs.compute_match;
  case Generic:
    if (body_op_name != rhs.body_op_name)
      return body_op_name < rhs.body_op_name;
    if (generic_class != rhs.generic_class)
      return generic_class < rhs.generic_class;
    // Preserve the legacy Generic matching contract for unannotated/default
    // kernels: compound kernel generics and single-body hw_spec generics can
    // have different DPS arities. Any non-default signature remains fully
    // exact.
    if (hasOnlyDefaultMemoryOperands(compute_match) &&
        hasOnlyDefaultMemoryOperands(rhs.compute_match))
      return false;
    return compute_match < rhs.compute_match;
  case DataMover:
    if (data_mover_kind != rhs.data_mover_kind)
      return data_mover_kind < rhs.data_mover_kind;
    if (src_mem_space != rhs.src_mem_space)
      return src_mem_space < rhs.src_mem_space;
    if (dst_mem_space != rhs.dst_mem_space)
      return dst_mem_space < rhs.dst_mem_space;
    if (src_mem_kind != rhs.src_mem_kind)
      return src_mem_kind < rhs.src_mem_kind;
    if (dst_mem_kind != rhs.dst_mem_kind)
      return dst_mem_kind < rhs.dst_mem_kind;
    return broadcast < rhs.broadcast;
  }
  llvm_unreachable("unknown HWOpKey kind");
}

HWOpKey HWOpKey::named(std::string op_name, ComputeOpMatchInfo match) {
  HWOpKey k;
  k.kind = Named;
  k.linalg_op_name = std::move(op_name);
  k.compute_match = std::move(match);
  return k;
}

HWOpKey HWOpKey::generic(std::string body_op, GenericClass cls,
                         ComputeOpMatchInfo match) {
  HWOpKey k;
  k.kind = Generic;
  k.body_op_name = std::move(body_op);
  k.generic_class = cls;
  k.compute_match = std::move(match);
  return k;
}

HWOpKey HWOpKey::dataMover(DataMoverKind kind, std::string src, std::string dst,
                           std::optional<int64_t> src_kind,
                           std::optional<int64_t> dst_kind,
                           std::vector<int64_t> bcast) {
  HWOpKey k;
  k.kind = DataMover;
  k.data_mover_kind = kind;
  k.src_mem_space = std::move(src);
  k.dst_mem_space = std::move(dst);
  k.src_mem_kind = src_kind;
  k.dst_mem_kind = dst_kind;
  k.broadcast = std::move(bcast);
  return k;
}

// ============================================================
// Shared helper: collect loom.bind_shape bindings
// ============================================================

llvm::DenseMap<mlir::Value, HWTensorBinding>
HWOpRegistry::collectBindingMap(mlir::func::FuncOp func) {
  llvm::DenseMap<mlir::Value, HWTensorBinding> bindingMap;
  func.walk([&](loom::BindShapeOp bindshapeOp) {
    HWTensorBinding binding;
    for (mlir::Value sym : bindshapeOp.getSymbols()) {
      llvm::StringRef symName = loom::utils::traceToSymbolicVar(sym);
      binding.dim_symbols.push_back(
          symName.empty() ? "?" : std::string(symName));
    }
    bindingMap[bindshapeOp.getMemref()] = std::move(binding);
  });
  return bindingMap;
}

// ============================================================
// Resource Map
// ============================================================

void HWOpRegistry::buildResourceMap(mlir::ModuleOp platformModule) {
  for (mlir::Operation &op : platformModule.getBody()->getOperations()) {
    auto extractResources = [&](llvm::StringRef moduleName,
                                mlir::Operation::operand_range resources) {
      auto &res = module_resource_map_[moduleName.str()];
      for (mlir::Value resourceVal : resources) {
        if (auto ResourceExclusiveOp =
                resourceVal.getDefiningOp<mlir::adl::ResourceExclusiveOp>()) {
          res.push_back(ResourceExclusiveOp.getSymName().str());
        }
      }
    };

    if (auto computeOp = llvm::dyn_cast<mlir::adl::ProcessorComputeOp>(&op)) {
      extractResources(computeOp.getSymName(), computeOp.getResources());
    } else if (auto dmoverOp = llvm::dyn_cast<mlir::adl::ProcessorDMoverOp>(&op)) {
      extractResources(dmoverOp.getSymName(), dmoverOp.getResources());
    }
  }
}

// ============================================================
// Loading
// ============================================================

mlir::LogicalResult
HWOpRegistry::loadFromPlatformFile(llvm::StringRef file_path,
                                   mlir::MLIRContext &context) {
  llvm::SourceMgr sm;
  auto file = mlir::openInputFile(file_path);
  if (!file) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to open platform IR file: " << file_path << "\n";
    return mlir::failure();
  }
  sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sm, &context);
  if (!module) {
    llvm::WithColor::error(llvm::errs())
        << "Failed to parse platform IR file: " << file_path << "\n";
    return mlir::failure();
  }

  // Build resource map from processor ops before walking sub-modules.
  buildResourceMap(*module);

  // Walk immediate children for sub-modules.
  for (mlir::Operation &op : module->getBody()->getOperations()) {
    auto subModule = llvm::dyn_cast<mlir::ModuleOp>(&op);
    if (!subModule)
      continue;

    llvm::StringRef component =
        subModule.getName().value_or(llvm::StringRef(""));
    if (component.empty())
      continue;
    llvm::StringRef processorArray = component;
    llvm::StringRef processorDefinition;
    std::vector<std::string> processorDomain;
    if (auto attr = subModule->getAttrOfType<mlir::StringAttr>(
            "mlar.processor_array"))
      processorArray = attr.getValue();
    if (auto attr = subModule->getAttrOfType<mlir::StringAttr>(
            "mlar.processor_definition"))
      processorDefinition = attr.getValue();
    if (auto attr =
            subModule->getAttrOfType<mlir::ArrayAttr>("mlar.processor_domain"))
      for (mlir::Attribute axis : attr) {
        auto name = mlir::dyn_cast<mlir::StringAttr>(axis);
        if (!name) {
          subModule.emitError()
              << "mlar.processor_domain must contain only strings";
          return mlir::failure();
        }
        processorDomain.push_back(name.getValue().str());
      }

    // Detect data mover module: any func containing a loom.copy or loom.gather.
    bool is_data_mover = false;
    subModule.walk([&](loom::CopyOp) { is_data_mover = true; });
    subModule.walk([&](loom::GatherOp) { is_data_mover = true; });

    if (mlir::failed(indexModule(subModule, component, processorArray,
                                 processorDefinition,
                                 std::move(processorDomain), is_data_mover)))
      return mlir::failure();
  }

  platform_module_ = std::move(module);
  return mlir::success();
}

// ============================================================
// Lookup
// ============================================================

const HWComputeFunc *HWOpRegistry::lookup(const HWOpKey &key) const {
  auto it = registry_.find(key);
  if (it != registry_.end() && it->second.size() == 1)
    return &it->second.front();
  return nullptr;
}

std::vector<const HWComputeFunc *>
HWOpRegistry::lookupCandidates(const HWOpKey &key) const {
  std::vector<const HWComputeFunc *> result;
  auto it = registry_.find(key);
  if (it == registry_.end())
    return result;
  result.reserve(it->second.size());
  for (const HWComputeFunc &candidate : it->second)
    result.push_back(&candidate);
  return result;
}

std::vector<const HWComputeFunc *>
HWOpRegistry::lookupComputeCandidates(const HWOpKey &semantic_key) const {
  std::vector<const HWComputeFunc *> result;
  if (semantic_key.kind == HWOpKey::DataMover)
    return result;
  for (const auto &[key, implementations] : registry_) {
    bool matches = key.kind == semantic_key.kind;
    if (matches && key.kind == HWOpKey::Named)
      matches = key.linalg_op_name == semantic_key.linalg_op_name;
    if (matches && key.kind == HWOpKey::Generic)
      matches = key.body_op_name == semantic_key.body_op_name &&
                key.generic_class == semantic_key.generic_class;
    if (!matches)
      continue;
    for (const HWComputeFunc &implementation : implementations)
      result.push_back(&implementation);
  }
  llvm::sort(result, [](const HWComputeFunc *lhs, const HWComputeFunc *rhs) {
    return lhs->registration_order < rhs->registration_order;
  });
  return result;
}

const HWComputeFunc *
HWOpRegistry::lookupExact(llvm::StringRef processor_array,
                          llvm::StringRef function) const {
  auto identity = std::make_pair(processor_array.str(), function.str());
  if (!identities_.count(identity))
    return nullptr;
  for (const auto &[key, candidates] : registry_)
    for (const HWComputeFunc &candidate : candidates)
      if (candidate.hw_component == processor_array &&
          candidate.hw_func_name == function)
        return &candidate;
  for (const HWComputeFunc &candidate : symbolic_data_movers_)
    if (candidate.hw_component == processor_array &&
        candidate.hw_func_name == function)
      return &candidate;
  return nullptr;
}

const HWComputeFunc *HWOpRegistry::lookupDataMover(
    DataMoverKind kind, llvm::StringRef src_mem_space,
    llvm::StringRef dst_mem_space, std::optional<int64_t> src_mem_kind,
    std::optional<int64_t> dst_mem_kind, llvm::ArrayRef<int64_t> area) const {
  std::string src = detail::canonicalMemSpace(src_mem_space);
  std::string dst = detail::canonicalMemSpace(dst_mem_space);
  HWOpKey exact = HWOpKey::dataMover(kind, src, dst, src_mem_kind, dst_mem_kind,
                         std::vector<int64_t>(area.begin(), area.end()));
  if (const HWComputeFunc *hwFunc = lookup(exact))
    return hwFunc;

  if (kind == DataMoverKind::Copy && detail::isAllOnes(area))
    return nullptr;

  const HWComputeFunc *match = nullptr;
  for (const HWComputeFunc &candidate : symbolic_data_movers_) {
    if (candidate.data_mover_kind != kind ||
        candidate.src_mem_space != src || candidate.dst_mem_space != dst ||
        candidate.src_mem_kind != src_mem_kind ||
        candidate.dst_mem_kind != dst_mem_kind ||
        candidate.broadcast.size() != area.size())
      continue;

    bool matches = true;
    for (size_t i = 0; i < area.size(); ++i) {
      int64_t hwArea = candidate.broadcast[i];
      if (!mlir::ShapedType::isDynamic(hwArea) && hwArea != area[i]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      if (match)
        return nullptr;
      match = &candidate;
    }
  }
  return match;
}

std::vector<const HWComputeFunc *> HWOpRegistry::lookupDataMoverCandidates(
    DataMoverKind kind, llvm::StringRef src_mem_space,
    llvm::StringRef dst_mem_space, llvm::ArrayRef<int64_t> area) const {
  std::string src = detail::canonicalMemSpace(src_mem_space);
  std::string dst = detail::canonicalMemSpace(dst_mem_space);
  std::vector<const HWComputeFunc *> result;
  auto appendIfCompatible = [&](const HWComputeFunc &candidate) {
    if (!candidate.is_data_mover || candidate.data_mover_kind != kind ||
        candidate.src_mem_space != src || candidate.dst_mem_space != dst ||
        candidate.broadcast.size() != area.size())
      return;
    for (size_t i = 0; i < area.size(); ++i)
      if (!mlir::ShapedType::isDynamic(candidate.broadcast[i]) &&
          candidate.broadcast[i] != area[i])
        return;
    result.push_back(&candidate);
  };
  for (const auto &[key, implementations] : registry_)
    for (const HWComputeFunc &candidate : implementations)
      appendIfCompatible(candidate);
  for (const HWComputeFunc &candidate : symbolic_data_movers_)
    appendIfCompatible(candidate);
  return result;
}

std::vector<const HWComputeFunc *>
HWOpRegistry::lookupUnicastDataMoverCandidates(
    DataMoverKind kind, llvm::StringRef src_mem_space,
    llvm::StringRef dst_mem_space) const {
  std::string src = detail::canonicalMemSpace(src_mem_space);
  std::string dst = detail::canonicalMemSpace(dst_mem_space);
  std::vector<const HWComputeFunc *> result;
  auto appendIfCompatible = [&](const HWComputeFunc &candidate) {
    bool supportsUnicast = llvm::all_of(candidate.broadcast, [](int64_t area) {
      return area == 1 || mlir::ShapedType::isDynamic(area);
    });
    if (!candidate.is_data_mover || candidate.data_mover_kind != kind ||
        candidate.src_mem_space != src || candidate.dst_mem_space != dst ||
        !supportsUnicast)
      return;
    result.push_back(&candidate);
  };
  for (const auto &[key, implementations] : registry_)
    for (const HWComputeFunc &candidate : implementations)
      appendIfCompatible(candidate);
  for (const HWComputeFunc &candidate : symbolic_data_movers_)
    appendIfCompatible(candidate);
  return result;
}

// ============================================================
// Indexing
// ============================================================

mlir::LogicalResult HWOpRegistry::indexModule(mlir::ModuleOp module,
                                              llvm::StringRef module_symbol,
                                              llvm::StringRef processor_array,
                                              llvm::StringRef processor_definition,
                                              std::vector<std::string> processor_domain,
                                              bool is_data_mover) {
  // Look up resources for this module.
  std::vector<std::string> resources;
  auto resIt = module_resource_map_.find(module_symbol.str());
  if (resIt != module_resource_map_.end())
    resources = resIt->second;

  bool failed = false;
  module.walk([&](mlir::func::FuncOp func) {
    if (failed)
      return;
    std::optional<HWComputeFunc> hwFunc;
    if (is_data_mover) {
      hwFunc = extractDataMoverFromFunc(func, module_symbol, processor_array,
                                        processor_definition);
    } else {
      mlir::FailureOr<std::optional<HWComputeFunc>> extracted =
          extractFromFunc(func, module_symbol, processor_array,
                          processor_definition);
      if (mlir::failed(extracted)) {
        failed = true;
        return;
      }
      hwFunc = std::move(*extracted);
    }

    if (!hwFunc)
      return;

    // Attach resources from the processor declaration.
    hwFunc->resources = resources;
    hwFunc->processor_domain = processor_domain;
    hwFunc->registration_order = next_registration_order_++;

    // Build unified key and insert.
    HWOpKey key;
    auto identity =
        std::make_pair(hwFunc->hw_component, hwFunc->hw_func_name);
    if (identities_.count(identity)) {
      func.emitError() << "duplicate hardware implementation identity ('"
                       << identity.first << "', '" << identity.second << "')";
      failed = true;
      return;
    }

    if (hwFunc->is_data_mover) {
      if (detail::isSymbolicArea(hwFunc->broadcast)) {
        identities_.emplace(identity, HWOpKey::dataMover(
                                          hwFunc->data_mover_kind,
                                          hwFunc->src_mem_space,
                                          hwFunc->dst_mem_space,
                                          hwFunc->src_mem_kind,
                                          hwFunc->dst_mem_kind,
                                          hwFunc->broadcast));
        symbolic_data_movers_.push_back(std::move(*hwFunc));
        return;
      }
      key = HWOpKey::dataMover(hwFunc->data_mover_kind,
                               hwFunc->src_mem_space, hwFunc->dst_mem_space,
                               hwFunc->src_mem_kind, hwFunc->dst_mem_kind,
                               hwFunc->broadcast);
    } else if (!hwFunc->body_op_name.empty()) {
      key = HWOpKey::generic(hwFunc->body_op_name, hwFunc->generic_class,
                             hwFunc->compute_match);
    } else {
      key = HWOpKey::named(hwFunc->linalg_op_name, hwFunc->compute_match);
    }

    identities_.emplace(std::move(identity), key);
    registry_[key].push_back(std::move(*hwFunc));
  });
  return failed ? mlir::failure() : mlir::success();
}

} // namespace lcs
} // namespace loom
