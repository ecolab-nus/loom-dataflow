/**
 * @file hw_op_registry.cpp
 * @brief Implementation of HWOpRegistry for loading and indexing
 *        hardware platform IR files.
 */

#include "hw_op_registry.h"
#include "hw_op_registry_detail.h"
#include "utils.h"
#include "ADLDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"
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
    return linalg_op_name < rhs.linalg_op_name;
  case Generic:
    if (body_op_name != rhs.body_op_name)
      return body_op_name < rhs.body_op_name;
    return generic_class < rhs.generic_class;
  case DataMover:
    if (data_mover_kind != rhs.data_mover_kind)
      return data_mover_kind < rhs.data_mover_kind;
    if (src_mem_space != rhs.src_mem_space)
      return src_mem_space < rhs.src_mem_space;
    if (dst_mem_space != rhs.dst_mem_space)
      return dst_mem_space < rhs.dst_mem_space;
    return broadcast < rhs.broadcast;
  }
  llvm_unreachable("unknown HWOpKey kind");
}

HWOpKey HWOpKey::named(std::string op_name) {
  HWOpKey k;
  k.kind = Named;
  k.linalg_op_name = std::move(op_name);
  return k;
}

HWOpKey HWOpKey::generic(std::string body_op, GenericClass cls) {
  HWOpKey k;
  k.kind = Generic;
  k.body_op_name = std::move(body_op);
  k.generic_class = cls;
  return k;
}

HWOpKey HWOpKey::dataMover(DataMoverKind kind, std::string src, std::string dst,
                           std::vector<int64_t> bcast) {
  HWOpKey k;
  k.kind = DataMover;
  k.data_mover_kind = kind;
  k.src_mem_space = std::move(src);
  k.dst_mem_space = std::move(dst);
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
                resourceVal.getDefiningOp<adl::ResourceExclusiveOp>()) {
          res.push_back(ResourceExclusiveOp.getSymName().str());
        }
      }
    };

    if (auto computeOp = llvm::dyn_cast<adl::ProcessorComputeOp>(&op)) {
      extractResources(computeOp.getSymName(), computeOp.getResources());
    } else if (auto dmoverOp = llvm::dyn_cast<adl::ProcessorDMoverOp>(&op)) {
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

    // Detect data mover module: any func containing a loom.copy or loom.gather.
    bool is_data_mover = false;
    subModule.walk([&](loom::CopyOp) { is_data_mover = true; });
    subModule.walk([&](loom::GatherOp) { is_data_mover = true; });

    indexModule(subModule, component, is_data_mover);
  }

  platform_module_ = std::move(module);
  return mlir::success();
}

// ============================================================
// Lookup
// ============================================================

const HWComputeFunc *HWOpRegistry::lookup(const HWOpKey &key) const {
  auto it = registry_.find(key);
  if (it != registry_.end())
    return &it->second;
  return nullptr;
}

const HWComputeFunc *HWOpRegistry::lookupDataMover(
    DataMoverKind kind, llvm::StringRef src_mem_space,
    llvm::StringRef dst_mem_space, llvm::ArrayRef<int64_t> area) const {
  std::string src = detail::canonicalMemSpace(src_mem_space);
  std::string dst = detail::canonicalMemSpace(dst_mem_space);
  HWOpKey exact =
      HWOpKey::dataMover(kind, src, dst,
                         std::vector<int64_t>(area.begin(), area.end()));
  if (const HWComputeFunc *hwFunc = lookup(exact))
    return hwFunc;

  if (kind == DataMoverKind::Copy && detail::isAllOnes(area))
    return nullptr;

  for (const HWComputeFunc &candidate : symbolic_data_movers_) {
    if (candidate.data_mover_kind != kind ||
        candidate.src_mem_space != src || candidate.dst_mem_space != dst ||
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
    if (matches)
      return &candidate;
  }

  return nullptr;
}

HWComputeFunc HWOpRegistry::makePlaceholder(llvm::StringRef op_name,
                                             llvm::StringRef hw_component) {
  HWComputeFunc p;
  p.hw_func_name = ("__unregistered__:" + op_name).str();
  p.hw_component = hw_component.str();
  return p;
}

// ============================================================
// Indexing
// ============================================================

void HWOpRegistry::indexModule(mlir::ModuleOp module,
                               llvm::StringRef hw_component,
                               bool is_data_mover) {
  // Look up resources for this module.
  std::vector<std::string> resources;
  auto resIt = module_resource_map_.find(hw_component.str());
  if (resIt != module_resource_map_.end())
    resources = resIt->second;

  module.walk([&](mlir::func::FuncOp func) {
    std::optional<HWComputeFunc> hwFunc;
    if (is_data_mover) {
      hwFunc = extractDataMoverFromFunc(func, hw_component);
    } else {
      hwFunc = extractFromFunc(func, hw_component);
    }

    if (!hwFunc)
      return;

    // Attach resources from the processor declaration.
    hwFunc->resources = resources;

    // Build unified key and insert.
    HWOpKey key;
    if (hwFunc->is_data_mover) {
      if (detail::isSymbolicArea(hwFunc->broadcast)) {
        symbolic_data_movers_.push_back(std::move(*hwFunc));
        return;
      }
      key = HWOpKey::dataMover(hwFunc->data_mover_kind,
                               hwFunc->src_mem_space, hwFunc->dst_mem_space,
                               hwFunc->broadcast);
    } else if (!hwFunc->body_op_name.empty()) {
      key = HWOpKey::generic(hwFunc->body_op_name, hwFunc->generic_class);
    } else {
      key = HWOpKey::named(hwFunc->linalg_op_name);
    }

    registry_[key] = std::move(*hwFunc);
  });
}

} // namespace lcs
} // namespace loom
