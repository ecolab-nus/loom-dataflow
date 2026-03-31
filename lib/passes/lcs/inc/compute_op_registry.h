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
#include <tuple>
#include <vector>

namespace loom {
namespace lcs {

/// Per-tensor binding from hardware IR: symbol names for each dimension.
/// For `loom.bind %A, [%M, %K]`, stores dim_symbols = ["M", "K"].
struct HWTensorBinding {
  std::vector<std::string> dim_symbols;
};

/// A hardware function entry from a platform IR file.
struct HWComputeFunc {
  std::string linalg_op_name; // e.g., "linalg.matmul" (empty for data movers)
  std::string hw_func_name;   // e.g., "matmul_f16"
  std::string hw_component;   // e.g., "matrix_lane" (from sub-module name)
  std::string body_op_name;   // inner arith/math op (empty for named/data mover ops)
  GenericClass generic_class = GenericClass::Parallel;
  std::string parallel_symbol;  // hw symbol for folded parallel product
  std::string reduction_symbol; // hw symbol for folded reduction product
  std::vector<HWTensorBinding> input_bindings;
  std::vector<HWTensorBinding> output_bindings;

  // Data mover specific (empty for compute ops)
  bool is_data_mover = false;
  std::string src_mem_space;        // e.g., "DRAM"
  std::string dst_mem_space;        // e.g., "L1"
  std::vector<int64_t> broadcast;   // e.g., {1,1} or {8,8}
};

/// Registry that loads a platform IR file and indexes hardware functions
/// by linalg op type (compute) or by copy attributes (data movers).
class HWOpRegistry {
public:
  /// Load a single platform MLIR file, walk its sub-modules, and build the
  /// index. Each sub-module name becomes the hw_component identifier.
  mlir::LogicalResult loadFromPlatformFile(llvm::StringRef file_path,
                                           mlir::MLIRContext &context);

  /// Look up a named linalg op (matmul, batch_matmul, etc.).
  const HWComputeFunc *lookupMatrixOp(llvm::StringRef linalg_op_name) const;

  /// Look up a vector_lane generic op by body op name and generic class.
  const HWComputeFunc *lookupVectorOp(llvm::StringRef body_op_name,
                                       GenericClass cls) const;

  /// Look up a data mover op by its static attributes.
  const HWComputeFunc *lookupDataMoverOp(llvm::StringRef src_mem_space,
                                          llvm::StringRef dst_mem_space,
                                          llvm::ArrayRef<int64_t> broadcast) const;

  /// Create a placeholder entry for an unregistered operation.
  static HWComputeFunc makePlaceholder(llvm::StringRef op_name,
                                        llvm::StringRef hw_component = "__unregistered__");

  /// Return the parsed platform module (kept alive by this registry).
  mlir::ModuleOp getPlatformModule() const { return *platform_module_; }

private:
  /// Named ops keyed by linalg op name (e.g., "linalg.matmul")
  std::map<std::string, HWComputeFunc> matrix_registry_;

  /// Vector lane ops keyed by (body_op_name, GenericClass)
  std::map<std::pair<std::string, GenericClass>, HWComputeFunc> vector_registry_;

  /// Data movers keyed by (src_mem_space, dst_mem_space, broadcast)
  using DataMoverKey = std::tuple<std::string, std::string, std::vector<int64_t>>;
  std::map<DataMoverKey, HWComputeFunc> data_mover_registry_;

  /// Keep the parsed platform module alive for the lifetime of the registry.
  mlir::OwningOpRef<mlir::ModuleOp> platform_module_;

  /// Collect loom.bind_shape bindings from a function.
  static llvm::DenseMap<mlir::Value, HWTensorBinding>
  collectBindingMap(mlir::func::FuncOp func);

  /// Index all funcs in a module under the given hw_component name.
  /// If is_data_mover is true, routes to data mover extraction.
  void indexModule(mlir::ModuleOp module, llvm::StringRef hw_component,
                   bool is_data_mover);

  /// Extract HWComputeFunc from a single func.func with a linalg compute op.
  std::optional<HWComputeFunc>
  extractFromFunc(mlir::func::FuncOp func, llvm::StringRef hw_component);

  /// Extract HWComputeFunc from a single func.func with a loom.copy op.
  std::optional<HWComputeFunc>
  extractDataMoverFromFunc(mlir::func::FuncOp func,
                           llvm::StringRef hw_component);
};

} // namespace lcs
} // namespace loom

#endif // LOOM_LCS_COMPUTE_OP_REGISTRY_H
