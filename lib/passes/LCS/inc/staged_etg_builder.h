#ifndef LOOM_LCS_STAGED_ETG_BUILDER_H
#define LOOM_LCS_STAGED_ETG_BUILDER_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

/// Simple workload record: operation and its symbolic dimensions.
struct Workload {
  std::string op;
  std::string dims;

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
  std::string stage_time = "MAX(.queues[*].resolved_time)";

  Stage(int id);
  void pushWorkload(const std::string &unit_name, const std::string &op,
                    const std::string &dims);
  void dump(llvm::raw_ostream &os, int indent = 0) const;
  llvm::json::Value toJSON() const;
};

/// A scope is a collection of stages with a symbolic time expression.
/// Separates compute and memory operations into distinct scopes.
struct Scope {
  std::string scope_name;
  std::map<int, Stage> stages;
  std::string scope_time = "SUM(.stages[*].stage_time)";

  Scope(std::string name);
  Stage &getOrCreateStage(int id);
  void dump(llvm::raw_ostream &os, int indent = 0) const;
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

  VariantETG(llvm::StringRef name);

  /// Build ETG from an affine.for loop body.
  void buildFromAffineFor(mlir::affine::AffineForOp for_op);

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
