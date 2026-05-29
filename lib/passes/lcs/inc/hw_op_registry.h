#ifndef LOOM_LCS_HW_OP_REGISTRY_H
#define LOOM_LCS_HW_OP_REGISTRY_H

#include "lcs_utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace loom {
namespace lcs {

enum class DataMoverKind { Copy, Gather };

/// Per-tensor binding from hardware IR: symbol names for each dimension.
/// For `loom.bind %A, [%M, %K]`, stores dim_symbols = ["M", "K"].
struct HWTensorBinding {
  std::vector<std::string> dim_symbols;
};

/// Unified lookup key for all hardware operations (named, generic, data mover).
struct HWOpKey {
  enum Kind { Named, Generic, DataMover };
  Kind kind;

  // Named: linalg op name (e.g., "linalg.matmul", "linalg.add")
  std::string linalg_op_name;

  // Generic: body arith/math op name + iterator classification
  std::string body_op_name;
  GenericClass generic_class = GenericClass::Parallel;

  // DataMover: transfer attributes
  DataMoverKind data_mover_kind = DataMoverKind::Copy;
  std::string src_mem_space;
  std::string dst_mem_space;
  std::vector<int64_t> broadcast;

  bool operator<(const HWOpKey &rhs) const;

  static HWOpKey named(std::string op_name);
  static HWOpKey generic(std::string body_op, GenericClass cls);
  static HWOpKey dataMover(DataMoverKind kind, std::string src, std::string dst,
                           std::vector<int64_t> bcast);
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
  std::vector<std::string> resources; // e.g., {"FPU"} or {"L1_torus::h", "L1_torus::v"}

  // Data mover specific (empty for compute ops)
  bool is_data_mover = false;
  DataMoverKind data_mover_kind = DataMoverKind::Copy;
  std::string src_mem_space;        // e.g., "DRAM"
  std::string dst_mem_space;        // e.g., "L1"
  // Static entries are constants; dynamic entries use ShapedType::kDynamic and
  // carry the corresponding hardware symbol in area_symbols.
  std::vector<int64_t> broadcast;   // e.g., {1,1} or {?,?}
  std::vector<std::string> area_symbols;
};

/// Registry that loads a platform IR file and indexes hardware functions
/// by linalg op type (compute) or by copy attributes (data movers).
class HWOpRegistry {
public:
  /// Load a single platform MLIR file, walk its sub-modules, and build the
  /// index. Each sub-module name becomes the hw_component identifier.
  mlir::LogicalResult loadFromPlatformFile(llvm::StringRef file_path,
                                           mlir::MLIRContext &context);

  /// Unified lookup: find a registered hw func by key.
  const HWComputeFunc *lookup(const HWOpKey &key) const;

  /// Data-mover lookup with symbolic-area fallback.
  const HWComputeFunc *lookupDataMover(DataMoverKind kind,
                                       llvm::StringRef src_mem_space,
                                       llvm::StringRef dst_mem_space,
                                       llvm::ArrayRef<int64_t> area) const;

  /// Create a placeholder entry for an unregistered operation.
  static HWComputeFunc makePlaceholder(llvm::StringRef op_name,
                                        llvm::StringRef hw_component = "__unregistered__");

  /// Return the parsed platform module (kept alive by this registry).
  mlir::ModuleOp getPlatformModule() const { return *platform_module_; }

private:
  /// Unified registry keyed by HWOpKey.
  std::map<HWOpKey, HWComputeFunc> registry_;

  /// Symbolic-area data movers that cannot be represented by exact static key.
  std::vector<HWComputeFunc> symbolic_data_movers_;

  /// Maps processor module name -> resource names. Built once during load.
  std::map<std::string, std::vector<std::string>> module_resource_map_;

  /// Keep the parsed platform module alive for the lifetime of the registry.
  mlir::OwningOpRef<mlir::ModuleOp> platform_module_;

  /// Walk platform module to build module_resource_map_.
  void buildResourceMap(mlir::ModuleOp platformModule);

  /// Collect loom.bind_shape bindings from a function.
  static llvm::DenseMap<mlir::Value, HWTensorBinding>
  collectBindingMap(mlir::func::FuncOp func);

  /// Index all funcs in a module under the given hw_component name.
  /// If is_data_mover is true, routes to data mover extraction.
  void indexModule(mlir::ModuleOp module, llvm::StringRef hw_component,
                   bool is_data_mover);

  /// Find the unique non-infra linalg op in a func; nullptr if none or ambiguous.
  static mlir::Operation *findComputeOp(mlir::func::FuncOp func);

  /// Fill input_bindings and output_bindings on result from a LinalgOp.
  static void fillInputOutputBindings(
      mlir::linalg::LinalgOp linalgOp,
      const llvm::DenseMap<mlir::Value, HWTensorBinding> &bindingMap,
      HWComputeFunc &result);

  /// Fill generic-specific fields (body_op_name, generic_class, symbols).
  /// Returns false for compound ops (bodyOpCount != 1).
  static bool fillGenericDetails(
      mlir::Operation *computeOp,
      const llvm::DenseMap<mlir::Value, HWTensorBinding> &bindingMap,
      HWComputeFunc &result);

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

#endif // LOOM_LCS_HW_OP_REGISTRY_H
