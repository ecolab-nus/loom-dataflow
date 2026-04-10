/**
 * @file compute_op_registry.cpp
 * @brief Implementation of HWOpRegistry for loading and indexing
 *        hardware platform IR files.
 */

#include "compute_op_registry.h"
#include "utils.h"
#include "ADLDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
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

HWOpKey HWOpKey::dataMover(std::string src, std::string dst,
                           std::vector<int64_t> bcast) {
  HWOpKey k;
  k.kind = DataMover;
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

    // Detect data mover module: any func containing a loom.copy op.
    bool is_data_mover = false;
    subModule.walk([&](loom::CopyOp) { is_data_mover = true; });

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
      key = HWOpKey::dataMover(hwFunc->src_mem_space, hwFunc->dst_mem_space,
                               hwFunc->broadcast);
    } else if (!hwFunc->body_op_name.empty()) {
      key = HWOpKey::generic(hwFunc->body_op_name, hwFunc->generic_class);
    } else {
      key = HWOpKey::named(hwFunc->linalg_op_name);
    }

    registry_[key] = std::move(*hwFunc);
  });
}

// ============================================================
// Extraction helpers: compute ops (linalg-based)
// ============================================================

mlir::Operation *HWOpRegistry::findComputeOp(mlir::func::FuncOp func) {
  mlir::Operation *computeOp = nullptr;
  func.walk([&](mlir::linalg::LinalgOp linalgOp) {
    mlir::Operation *op = linalgOp.getOperation();
    if (llvm::isa<mlir::linalg::FillOp, mlir::linalg::CopyOp>(op))
      return;
    assert(!computeOp && "Expected exactly one compute linalg op per hw func");
    computeOp = op;
  });
  return computeOp;
}

void HWOpRegistry::fillInputOutputBindings(
    mlir::linalg::LinalgOp linalgOp,
    const llvm::DenseMap<mlir::Value, HWTensorBinding> &bindingMap,
    HWComputeFunc &result) {
  auto pushBinding = [&](mlir::Value val,
                         std::vector<HWTensorBinding> &bindings) {
    auto it = bindingMap.find(val);
    bindings.push_back(it != bindingMap.end() ? it->second : HWTensorBinding{});
  };
  for (mlir::Value input : linalgOp.getDpsInputs())
    pushBinding(input, result.input_bindings);
  for (mlir::Value output : linalgOp.getDpsInits())
    pushBinding(output, result.output_bindings);
}

bool HWOpRegistry::fillGenericDetails(
    mlir::Operation *computeOp,
    const llvm::DenseMap<mlir::Value, HWTensorBinding> &bindingMap,
    HWComputeFunc &result) {
  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(computeOp);

  // Find the single arith/math body op (compound ops like cmpf+select → skip).
  mlir::Operation *singleBodyOp = nullptr;
  unsigned bodyOpCount = 0;
  for (mlir::Operation &bodyOp : computeOp->getRegion(0).front()) {
    if (llvm::isa<mlir::linalg::YieldOp>(&bodyOp))
      continue;
    mlir::Dialect *dialect = bodyOp.getDialect();
    if (!dialect)
      continue;
    llvm::StringRef ns = dialect->getNamespace();
    if (ns != "arith" && ns != "math")
      continue;
    singleBodyOp = &bodyOp;
    bodyOpCount++;
  }
  if (bodyOpCount != 1)
    return false;

  result.body_op_name = singleBodyOp->getName().getStringRef().str();

  // Classify iterator types.
  auto iteratorTypes = linalgOp.getIteratorTypesArray();
  result.generic_class = classifyIteratorTypes(iteratorTypes);

  // Derive parallel_symbol and reduction_symbol from indexing maps + bindings.
  auto indexingMaps = linalgOp.getIndexingMapsArray();
  llvm::SmallVector<mlir::Value> allOperands;
  for (auto v : linalgOp.getDpsInputs())
    allOperands.push_back(v);
  for (auto v : linalgOp.getDpsInits())
    allOperands.push_back(v);

  for (unsigned di = 0; di < iteratorTypes.size(); ++di) {
    bool found = false;
    for (unsigned opIdx = 0; opIdx < indexingMaps.size() && !found; ++opIdx) {
      mlir::AffineMap map = indexingMaps[opIdx];
      auto bindIt = bindingMap.find(allOperands[opIdx]);
      if (bindIt == bindingMap.end())
        continue;
      const auto &binding = bindIt->second;
      for (unsigned r = 0; r < map.getNumResults() && !found; ++r) {
        auto affineDim = mlir::dyn_cast<mlir::AffineDimExpr>(map.getResult(r));
        if (!affineDim || affineDim.getPosition() != di)
          continue;
        if (r >= binding.dim_symbols.size())
          continue;
        const std::string &sym = binding.dim_symbols[r];
        if (iteratorTypes[di] == mlir::utils::IteratorType::parallel &&
            result.parallel_symbol.empty())
          result.parallel_symbol = sym;
        else if (iteratorTypes[di] == mlir::utils::IteratorType::reduction &&
                 result.reduction_symbol.empty())
          result.reduction_symbol = sym;
        found = true;
      }
    }
  }
  return true;
}

// ============================================================
// Extraction: compute ops (linalg-based)
// ============================================================

std::optional<HWComputeFunc>
HWOpRegistry::extractFromFunc(mlir::func::FuncOp func,
                              llvm::StringRef hw_component) {
  auto bindingMap = collectBindingMap(func);

  mlir::Operation *computeOp = findComputeOp(func);
  if (!computeOp)
    return std::nullopt;

  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(computeOp);
  HWComputeFunc result;
  result.linalg_op_name = computeOp->getName().getStringRef().str();
  result.hw_func_name   = func.getName().str();
  result.hw_component   = hw_component.str();
  fillInputOutputBindings(linalgOp, bindingMap, result);

  if (llvm::isa<mlir::linalg::GenericOp>(computeOp))
    if (!fillGenericDetails(computeOp, bindingMap, result))
      return std::nullopt;

  return result;
}

// ============================================================
// Extraction: data mover ops (loom.copy-based)
// ============================================================

std::optional<HWComputeFunc>
HWOpRegistry::extractDataMoverFromFunc(mlir::func::FuncOp func,
                                       llvm::StringRef hw_component) {
  // 1. Collect loom.bind_shape bindings.
  auto bindingMap = collectBindingMap(func);

  // 2. Find the unique loom.copy op.
  loom::CopyOp copyOp = nullptr;
  func.walk([&](loom::CopyOp op) {
    assert(!copyOp && "Expected exactly one loom.copy per data mover func");
    copyOp = op;
  });

  if (!copyOp)
    return std::nullopt;

  HWComputeFunc result;
  result.is_data_mover = true;
  result.hw_func_name = func.getName().str();
  result.hw_component = hw_component.str();

  // Extract key attributes
  if (auto attr = copyOp.getSrcMemSpaceAttr())
    result.src_mem_space = attr.getLeafReference().str();
  if (auto attr = copyOp.getDstMemSpaceAttr())
    result.dst_mem_space = attr.getLeafReference().str();

  if (auto broadcastAttr = copyOp.getBroadcastAttr()) {
    for (auto val : broadcastAttr) {
      result.broadcast.push_back(
          mlir::cast<mlir::IntegerAttr>(val).getInt());
    }
  }

  // Source binding (input)
  auto srcIt = bindingMap.find(copyOp.getSource());
  if (srcIt != bindingMap.end()) {
    result.input_bindings.push_back(srcIt->second);
  } else {
    result.input_bindings.push_back(HWTensorBinding{});
  }

  // Destination binding (output)
  auto dstIt = bindingMap.find(copyOp.getDestination());
  if (dstIt != bindingMap.end()) {
    result.output_bindings.push_back(dstIt->second);
  } else {
    result.output_bindings.push_back(HWTensorBinding{});
  }

  return result;
}

} // namespace lcs
} // namespace loom
