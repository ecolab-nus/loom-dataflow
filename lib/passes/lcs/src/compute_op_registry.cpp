/**
 * @file compute_op_registry.cpp
 * @brief Implementation of HWOpRegistry for loading and indexing
 *        hardware platform IR files.
 */

#include "compute_op_registry.h"
#include "utils.h"
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

const HWComputeFunc *
HWOpRegistry::lookupMatrixOp(llvm::StringRef linalg_op_name) const {
  auto it = matrix_registry_.find(linalg_op_name.str());
  if (it != matrix_registry_.end())
    return &it->second;
  return nullptr;
}

const HWComputeFunc *
HWOpRegistry::lookupVectorOp(llvm::StringRef body_op_name,
                              GenericClass cls) const {
  auto key = std::make_pair(body_op_name.str(), cls);
  auto it = vector_registry_.find(key);
  if (it != vector_registry_.end())
    return &it->second;
  return nullptr;
}

const HWComputeFunc *
HWOpRegistry::lookupDataMoverOp(llvm::StringRef src_mem_space,
                                 llvm::StringRef dst_mem_space,
                                 llvm::ArrayRef<int64_t> broadcast) const {
  DataMoverKey key{src_mem_space.str(), dst_mem_space.str(),
                   std::vector<int64_t>(broadcast.begin(), broadcast.end())};
  auto it = data_mover_registry_.find(key);
  if (it != data_mover_registry_.end())
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
  module.walk([&](mlir::func::FuncOp func) {
    if (is_data_mover) {
      if (auto hwFunc = extractDataMoverFromFunc(func, hw_component)) {
        DataMoverKey key{hwFunc->src_mem_space, hwFunc->dst_mem_space,
                         hwFunc->broadcast};
        data_mover_registry_[key] = std::move(*hwFunc);
      }
    } else {
      if (auto hwFunc = extractFromFunc(func, hw_component)) {
        if (!hwFunc->body_op_name.empty()) {
          auto key =
              std::make_pair(hwFunc->body_op_name, hwFunc->generic_class);
          vector_registry_[key] = std::move(*hwFunc);
        } else {
          matrix_registry_[hwFunc->linalg_op_name] = std::move(*hwFunc);
        }
      }
    }
  });
}

// ============================================================
// Extraction: compute ops (linalg-based)
// ============================================================

std::optional<HWComputeFunc>
HWOpRegistry::extractFromFunc(mlir::func::FuncOp func,
                              llvm::StringRef hw_component) {
  // 1. Collect loom.bind_shape bindings.
  auto bindingMap = collectBindingMap(func);

  // 2. Find the unique compute linalg op (skip FillOp, CopyOp).
  mlir::Operation *computeOp = nullptr;
  func.walk([&](mlir::linalg::LinalgOp linalgOp) {
    mlir::Operation *op = linalgOp.getOperation();
    if (llvm::isa<mlir::linalg::FillOp, mlir::linalg::CopyOp>(op))
      return;
    assert(!computeOp &&
           "Expected exactly one compute linalg op per hw func");
    computeOp = op;
  });

  if (!computeOp)
    return std::nullopt;

  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(computeOp);
  HWComputeFunc result;
  result.linalg_op_name = computeOp->getName().getStringRef().str();
  result.hw_func_name = func.getName().str();
  result.hw_component = hw_component.str();

  // Input bindings
  for (mlir::Value input : linalgOp.getDpsInputs()) {
    auto it = bindingMap.find(input);
    if (it != bindingMap.end()) {
      result.input_bindings.push_back(it->second);
    } else {
      result.input_bindings.push_back(HWTensorBinding{});
    }
  }

  // Output bindings
  for (mlir::Value output : linalgOp.getDpsInits()) {
    auto it = bindingMap.find(output);
    if (it != bindingMap.end()) {
      result.output_bindings.push_back(it->second);
    } else {
      result.output_bindings.push_back(HWTensorBinding{});
    }
  }

  // 3. For linalg.generic: extract body op, classify, derive symbols.
  if (llvm::isa<mlir::linalg::GenericOp>(computeOp)) {
    // Walk the body block, count arith/math ops (skip linalg.yield)
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

    // Compound hw funcs (e.g., vec_max1_f16 with cmpf+select): skip
    if (bodyOpCount != 1)
      return std::nullopt;

    result.body_op_name = singleBodyOp->getName().getStringRef().str();

    // Classify via iterator_types
    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    bool hasPar = false, hasRed = false;
    for (auto it : iteratorTypes) {
      if (it == mlir::utils::IteratorType::parallel)
        hasPar = true;
      else if (it == mlir::utils::IteratorType::reduction)
        hasRed = true;
    }
    if (hasPar && hasRed)
      result.generic_class = GenericClass::Mixed;
    else if (hasRed)
      result.generic_class = GenericClass::Reduction;
    else
      result.generic_class = GenericClass::Parallel;

    // Derive parallel_symbol and reduction_symbol from indexing maps + bindings
    auto indexingMaps = linalgOp.getIndexingMapsArray();
    // Collect all operands in ins→outs order
    llvm::SmallVector<mlir::Value> allOperands;
    for (auto v : linalgOp.getDpsInputs())
      allOperands.push_back(v);
    for (auto v : linalgOp.getDpsInits())
      allOperands.push_back(v);

    for (unsigned di = 0; di < iteratorTypes.size(); ++di) {
      bool found = false;
      for (unsigned opIdx = 0; opIdx < indexingMaps.size(); ++opIdx) {
        mlir::AffineMap map = indexingMaps[opIdx];
        // Look up binding for this operand
        auto bindIt = bindingMap.find(allOperands[opIdx]);
        if (bindIt == bindingMap.end())
          continue;
        const auto &binding = bindIt->second;

        for (unsigned r = 0; r < map.getNumResults(); ++r) {
          auto affineDim =
              mlir::dyn_cast<mlir::AffineDimExpr>(map.getResult(r));
          if (affineDim && affineDim.getPosition() == di) {
            if (r < binding.dim_symbols.size()) {
              const std::string &sym = binding.dim_symbols[r];
              if (iteratorTypes[di] == mlir::utils::IteratorType::parallel &&
                  result.parallel_symbol.empty()) {
                result.parallel_symbol = sym;
              } else if (iteratorTypes[di] ==
                             mlir::utils::IteratorType::reduction &&
                         result.reduction_symbol.empty()) {
                result.reduction_symbol = sym;
              }
              found = true;
              break;
            }
          }
        }
        if (found)
          break;
      }
    }
  }

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
