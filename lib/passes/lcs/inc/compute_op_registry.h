#ifndef LOOM_LCS_COMPUTE_OP_REGISTRY_H
#define LOOM_LCS_COMPUTE_OP_REGISTRY_H

#include "lcs_utils.h"
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
struct HWComputeFunc {
  std::string linalg_op_name; // e.g., "linalg.matmul"
  std::string hw_func_name;   // e.g., "matmul_f16"
  std::string hw_component;   // e.g., "matrix_lane" (from filename stem)
  std::string body_op_name;   // inner arith/math op (empty for named ops)
  GenericClass generic_class = GenericClass::Parallel;
  std::string parallel_symbol;  // hw symbol for folded parallel product
  std::string reduction_symbol; // hw symbol for folded reduction product
  std::vector<HWTensorBinding> input_bindings;
  std::vector<HWTensorBinding> output_bindings;
};

/// Registry that loads hardware IR files and indexes them by linalg op type.
class ComputeOpRegistry {
public:
  /// Load all .mlir files from dir_path, parse them, and build the index.
  mlir::LogicalResult loadFromDirectory(llvm::StringRef dir_path,
                                        mlir::MLIRContext &context);

  /// Look up a named linalg op (matmul, batch_matmul, etc.).
  const HWComputeFunc *lookupMatrixOp(llvm::StringRef linalg_op_name) const;

  /// Look up a vector_lane generic op by body op name and generic class.
  const HWComputeFunc *lookupVectorOp(llvm::StringRef body_op_name,
                                       GenericClass cls) const;

private:
  /// Named ops keyed by linalg op name (e.g., "linalg.matmul")
  std::map<std::string, HWComputeFunc> matrix_registry_;

  /// Vector lane ops keyed by (body_op_name, GenericClass)
  std::map<std::pair<std::string, GenericClass>, HWComputeFunc> vector_registry_;

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
