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

    // Derive hw_component from filename stem (e.g., "matrix_lane.mlir" ->
    // "matrix_lane")
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
ComputeOpRegistry::lookup(llvm::StringRef linalg_op_name) const {
  auto it = registry_.find(linalg_op_name.str());
  if (it != registry_.end())
    return &it->second;
  return nullptr;
}

void ComputeOpRegistry::indexModule(mlir::ModuleOp module,
                                    llvm::StringRef hw_component) {
  module.walk([&](mlir::func::FuncOp func) {
    if (auto hwFunc = extractFromFunc(func, hw_component)) {
      registry_[hwFunc->linalg_op_name] = std::move(*hwFunc);
    }
  });
}

std::optional<HWComputeFunc>
ComputeOpRegistry::extractFromFunc(mlir::func::FuncOp func,
                                   llvm::StringRef hw_component) {
  // TODO: implement vector_lane op matching — skip for now since multiple
  // generic ops map to the same linalg.generic key and need disambiguation
  // by inner body ops.
  if (hw_component == "vector_lane") {
    return std::nullopt;
  }

  // 1. Collect loom.bind operations to build tensor -> symbol binding map.
  //    Key: the tensor SSA value, Value: HWTensorBinding with dim symbol names.
  llvm::DenseMap<mlir::Value, HWTensorBinding> bindingMap;

  func.walk([&](loom::BindOp bindOp) {
    HWTensorBinding binding;
    for (mlir::Value sym : bindOp.getSymbols()) {
      llvm::StringRef symName = loom::utils::traceToSymbolicVar(sym);
      binding.dim_symbols.push_back(
          symName.empty() ? "?" : std::string(symName));
    }
    bindingMap[bindOp.getTensor()] = std::move(binding);
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

  // 3. Build HWComputeFunc from the linalg op and binding map.
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
      // No binding found — push empty binding
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

  return result;
}

} // namespace lcs
} // namespace loom
