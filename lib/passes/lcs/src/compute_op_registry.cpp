/**
 * @file compute_op_registry.cpp
 * @brief Implementation of ComputeOpRegistry for loading and indexing
 *        hardware compute IR files.
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

mlir::LogicalResult
ComputeOpRegistry::loadFromDirectory(llvm::StringRef dir_path,
                                     mlir::MLIRContext &context) {
  std::error_code ec;
  for (llvm::sys::fs::directory_iterator dir(dir_path, ec), end;
       dir != end && !ec; dir.increment(ec)) {
    llvm::StringRef path = dir->path();
    if (!path.ends_with(".mlir"))
      continue;

    llvm::StringRef stem = llvm::sys::path::stem(path);

    llvm::SourceMgr sm;
    auto file = mlir::openInputFile(path);
    if (!file) {
      llvm::WithColor::error(llvm::errs())
          << "Failed to open hw IR file: " << path << "\n";
      return mlir::failure();
    }
    sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(sm, &context);
    if (!module) {
      llvm::WithColor::error(llvm::errs())
          << "Failed to parse hw IR file: " << path << "\n";
      return mlir::failure();
    }

    indexModule(*module, stem);
    modules_.push_back(std::move(module));
  }

  if (ec) {
    llvm::WithColor::error(llvm::errs())
        << "Error reading directory: " << ec.message() << "\n";
    return mlir::failure();
  }

  return mlir::success();
}

const HWComputeFunc *
ComputeOpRegistry::lookupMatrixOp(llvm::StringRef linalg_op_name) const {
  auto it = matrix_registry_.find(linalg_op_name.str());
  if (it != matrix_registry_.end())
    return &it->second;
  return nullptr;
}

const HWComputeFunc *
ComputeOpRegistry::lookupVectorOp(llvm::StringRef body_op_name,
                                   GenericClass cls) const {
  auto key = std::make_pair(body_op_name.str(), cls);
  auto it = vector_registry_.find(key);
  if (it != vector_registry_.end())
    return &it->second;
  return nullptr;
}

void ComputeOpRegistry::indexModule(mlir::ModuleOp module,
                                    llvm::StringRef hw_component) {
  module.walk([&](mlir::func::FuncOp func) {
    if (auto hwFunc = extractFromFunc(func, hw_component)) {
      if (!hwFunc->body_op_name.empty()) {
        auto key =
            std::make_pair(hwFunc->body_op_name, hwFunc->generic_class);
        vector_registry_[key] = std::move(*hwFunc);
      } else {
        matrix_registry_[hwFunc->linalg_op_name] = std::move(*hwFunc);
      }
    }
  });
}

std::optional<HWComputeFunc>
ComputeOpRegistry::extractFromFunc(mlir::func::FuncOp func,
                                   llvm::StringRef hw_component) {
  // 1. Collect loom.bind_shape operations to build tensor -> symbol binding map.
  llvm::DenseMap<mlir::Value, HWTensorBinding> bindingMap;

  func.walk([&](loom::BindShapeOp bindshapeOp) {
    HWTensorBinding binding;
    for (mlir::Value sym : bindshapeOp.getSymbols()) {
      llvm::StringRef symName = loom::utils::traceToSymbolicVar(sym);
      binding.dim_symbols.push_back(
          symName.empty() ? "?" : std::string(symName));
    }
    bindingMap[bindshapeOp.getTensor()] = std::move(binding);
  });

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

} // namespace lcs
} // namespace loom
