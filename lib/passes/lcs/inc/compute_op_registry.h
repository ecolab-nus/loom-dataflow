#ifndef LOOM_LCS_COMPUTE_OP_REGISTRY_H
#define LOOM_LCS_COMPUTE_OP_REGISTRY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/LogicalResult.h"
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace loom {
namespace lcs {

/// Per-tensor binding from hardware IR: symbol names for each dimension.
/// For `loom.bind %A, [%M, %K]`, stores dim_symbols = ["M", "K"].
struct HWTensorBinding {
  std::vector<std::string> dim_symbols;
};

/// A hardware compute function entry from a hw IR file.
/// Records the linalg op type, the hw func name, the hw component (filename
/// stem), and the per-tensor dimension symbol bindings.
struct HWComputeFunc {
  std::string linalg_op_name; // e.g., "linalg.matmul"
  std::string hw_func_name;   // e.g., "matmul_f16"
  std::string hw_component;   // e.g., "matrix_lane" (from filename stem)
  std::vector<HWTensorBinding> input_bindings;
  std::vector<HWTensorBinding> output_bindings;
};

/// Registry that loads hardware IR files and indexes them by linalg op type.
class ComputeOpRegistry {
public:
  /// Load all .mlir files from dir_path, parse them, and build the index.
  /// The MLIRContext must already have all necessary dialects loaded.
  mlir::LogicalResult loadFromDirectory(llvm::StringRef dir_path,
                                        mlir::MLIRContext &context);

  /// Look up a hardware compute function by linalg op name.
  /// Returns nullptr if no match found.
  const HWComputeFunc *lookup(llvm::StringRef linalg_op_name) const;

private:
  /// Indexed by linalg op name (e.g., "linalg.matmul")
  std::map<std::string, HWComputeFunc> registry_;

  /// Keep parsed modules alive for the lifetime of the registry.
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules_;

  /// Index all funcs in a module under the given hw_component name.
  void indexModule(mlir::ModuleOp module, llvm::StringRef hw_component);

  /// Extract HWComputeFunc from a single func.func in hardware IR.
  std::optional<HWComputeFunc>
  extractFromFunc(mlir::func::FuncOp func, llvm::StringRef hw_component);
};

} // namespace lcs
} // namespace loom

#endif // LOOM_LCS_COMPUTE_OP_REGISTRY_H
