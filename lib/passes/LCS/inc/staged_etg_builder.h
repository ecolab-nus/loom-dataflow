#ifndef LOOM_LCS_STAGED_ETG_BUILDER_H
#define LOOM_LCS_STAGED_ETG_BUILDER_H

#include "constraint_expr.h"
#include "expr.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace loom {
namespace lcs {

/// Simple workload record: operation name and its symbolic dimensions.
/// dims holds one Expr per tensor input (e.g., two entries for matmul).
struct Workload {
  std::string op;
  std::vector<Expr> dims;

  llvm::json::Value toJSON() const;
};

/// Hardware queue containing a list of workloads.
/// resolved_time is a placeholder for future time resolution.
struct HardwareQueue {
  std::string unit_name;
  std::vector<Workload> workloads;
  std::optional<std::string> resolved_time;

  void dump(llvm::raw_ostream &os, int indent = 0) const;
  llvm::json::Value toJSON() const;
};

/// A stage groups hardware queues with a symbolic time expression.
/// Stages represent a level of computation/memory pipeline hierarchy.
struct Stage {
  int stage_id;
  std::map<std::string, HardwareQueue> queues;

  Stage(int id);
  void pushWorkload(const std::string &unit_name, const std::string &op,
                    std::vector<Expr> dims);
  void dump(llvm::raw_ostream &os, int indent = 0) const;
  llvm::json::Value toJSON() const;
};

/// A scope is a collection of stages.
/// Separates compute and memory operations into distinct scopes.
struct Scope {
  std::string scope_name;
  std::map<int, Stage> stages;

  Scope(std::string name);
  Stage &getOrCreateStage(int id);
  void dump(llvm::raw_ostream &os, int indent = 0) const;
  llvm::json::Value toJSON() const;
};

/// ConstraintScope: Captures constraint metadata from a computation variant.
/// Contains symbolic block sizes, loop iteration counts, and assembly formulas.
struct ConstraintScope {
  // metadata.symbols: maps symbol name (e.g., "BB", "BM") to type ("int"/"bool")
  std::map<std::string, std::string> symbols;
  // metadata.L1_footprint: symbolic size of each @L1 allocation
  std::vector<Expr> l1_footprint;
  // metadata.datatype: element type shared by all @L1 allocations (e.g., "f32")
  std::string datatype;
  // metadata.iter_num.seq_iter: symbolic trip count of the sequential loop
  Expr seq_iter;
  // metadata.iter_num.temp_iter: symbolic trip counts of temporal loops
  std::vector<Expr> temp_iter;
  // hard_constraints: constraints that every valid block-size assignment must satisfy
  std::vector<ConstraintExpr> hard_constraints;

  llvm::json::Value toJSON() const;
};

/// VariantETG: Execution Task Graph builder for a computation variant.
/// Analyzes affine.for loops to extract compute and memory workloads,
/// organizing them into compute and memory scopes with stage-based scheduling.
class VariantETG {
public:
  std::string variant_name;
  Scope compute_scope;
  Scope memory_scope;
  ConstraintScope constraint_scope;

  VariantETG(llvm::StringRef name);

  /// Build ETG from an affine.for loop body.
  void buildFromAffineFor(mlir::affine::AffineForOp for_op);

  /// Build constraint scope from a func operation.
  /// Extracts symbolic block sizes and loop iteration counts.
  void buildConstraintScope(mlir::func::FuncOp func_op);

  void dump(llvm::raw_ostream &os) const;
  llvm::json::Value toJSON() const;

private:
  void dispatchToComputeQueues(mlir::Operation *op, Stage &target_stage);
  void dispatchToMemoryQueues(mlir::Operation *op, Stage &target_stage);
  static std::string classifyCopyTransfer(mlir::Operation *op);
};

} // namespace lcs
} // namespace loom

#endif // LOOM_LCS_STAGED_ETG_BUILDER_H
