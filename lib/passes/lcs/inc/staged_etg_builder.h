#ifndef LOOM_LCS_STAGED_ETG_BUILDER_H
#define LOOM_LCS_STAGED_ETG_BUILDER_H

#include "constraint_expr.h"
#include "expr.h"
#include "l1_footprint_estimator.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace lcs {

// Forward declarations
class HWOpRegistry;
class WorkloadStageBody;

/// Workload record: operation name, symbolic dimensions, and resource usage.
/// dims maps hardware symbol names to operator IR symbolic expressions.
struct Workload {
  std::string op;
  std::map<std::string, Expr> dims;
  std::vector<std::string> resources;

  llvm::json::Value toJSON() const;
};

/// Hardware queue containing a list of workloads.
struct HardwareQueue {
  std::string unit_name;
  std::vector<Workload> workloads;

  void dump(llvm::raw_ostream &os, int indent = 0) const;
};

// =============================================================================
// Stage hierarchy
// =============================================================================

/// Polymorphic body of a Stage. A stage is either a workload bundle (the
/// existing Parallel/Sequential form) or a nested for_loop_block.
class StageBody {
public:
  enum class Kind { Workload, ForLoop };

  virtual ~StageBody() = default;

  /// Discriminator — used in lieu of dynamic_cast (the project builds with
  /// -fno-rtti).
  virtual Kind getKind() const = 0;

  /// Returns the single-key object that represents this body, e.g.
  /// `{"Parallel": [...]}` or `{"for_loop_block": {...}}`. Stage::toJSON()
  /// adds the `stage_id` field around it.
  virtual llvm::json::Object toJSONFragment() const = 0;

  virtual void dump(llvm::raw_ostream &os, int indent) const = 0;
};

/// Workload-stage body: the historical "Parallel of Sequential" form. Holds
/// hardware queues; lazily groups workloads by resource overlap when emitting.
class WorkloadStageBody : public StageBody {
public:
  Kind getKind() const override { return Kind::Workload; }

  void pushWorkload(const std::string &unit_name, const std::string &op,
                    std::map<std::string, Expr> dims,
                    std::vector<std::string> resources);

  llvm::json::Object toJSONFragment() const override;
  void dump(llvm::raw_ostream &os, int indent) const override;

  bool empty() const { return queues_.empty(); }

private:
  std::map<std::string, HardwareQueue> queues_;
};

/// A stage groups hardware queues (or a nested loop block) at a given
/// dependency depth. The `body` decides which JSON form is emitted.
struct Stage {
  int stage_id;
  std::unique_ptr<StageBody> body;

  explicit Stage(int id) : stage_id(id) {}
  Stage(int id, std::unique_ptr<StageBody> b)
      : stage_id(id), body(std::move(b)) {}

  // Move-only.
  Stage(Stage &&) = default;
  Stage &operator=(Stage &&) = default;
  Stage(const Stage &) = delete;
  Stage &operator=(const Stage &) = delete;

  void dump(llvm::raw_ostream &os, int indent = 0) const;
  llvm::json::Value toJSON() const;
};

/// A scope is a collection of stages keyed by stage_id.
/// Separates compute and memory operations into distinct scopes.
struct Scope {
  std::string scope_name;
  std::map<int, Stage> stages;

  explicit Scope(std::string name);

  /// Get (or create) a stage at `id` whose body is a WorkloadStageBody.
  /// Asserts that any pre-existing stage at `id` is of workload type.
  WorkloadStageBody &getOrCreateWorkloadStage(int id);

  /// Insert a stage with a non-workload body (e.g. a for_loop_block) at the
  /// first free id starting from `min_id`. Returns the chosen id.
  int placeStage(int min_id, std::unique_ptr<StageBody> body);

  void dump(llvm::raw_ostream &os, int indent = 0) const;
  llvm::json::Value toJSON() const;
};

// =============================================================================
// FusedOpBlock / KernelBlock / ForLoopBlockStageBody
// =============================================================================

/// Shared body shape: load, compute, and store scopes that hold the modelled
/// workloads of a fused operation. Used by both `KernelBlock` (variant root)
/// and `ForLoopBlockStageBody` (nested loop level).
struct FusedOpBlock {
  Scope load_scope;
  Scope compute_scope;
  Scope store_scope;

  FusedOpBlock()
      : load_scope("LoadScope"), compute_scope("ComputeScope"),
        store_scope("StoreScope") {}

  /// Emit `{"load_scope": ..., "compute_scope": ..., "store_scope": ...}`
  /// (without enclosing braces of any wrapping object).
  llvm::json::Object emitScopesJSON() const;
  void dumpScopes(llvm::raw_ostream &os, int indent) const;
};

/// Top-level container of a variant. Holds the kernel func's load, compute,
/// and store scopes; nested scf.for loops appear as `for_loop_block` stages
/// inside `compute_scope.stages`.
class KernelBlock {
public:
  FusedOpBlock body;

  llvm::json::Value toJSON() const;
  void dump(llvm::raw_ostream &os, int indent = 0) const;
};

/// Loop iteration tag carried by a for_loop_block. Spatial loops
/// (affine.parallel) are not modelled — they are walked through transparently.
enum class IterTypeTag { Sequential, Temporal };

/// A stage body that represents a nested scf.for level. Recursively contains
/// its own load/compute/store scopes through `body`.
class ForLoopBlockStageBody : public StageBody {
public:
  ForLoopBlockStageBody(std::string block_sym, IterTypeTag iter_type,
                        Expr trip_count);

  Kind getKind() const override { return Kind::ForLoop; }

  FusedOpBlock body;

  Scope &loadScope() { return body.load_scope; }
  Scope &computeScope() { return body.compute_scope; }
  Scope &storeScope() { return body.store_scope; }
  const Scope &loadScope() const { return body.load_scope; }
  const Scope &computeScope() const { return body.compute_scope; }
  const Scope &storeScope() const { return body.store_scope; }

  llvm::json::Object toJSONFragment() const override;
  void dump(llvm::raw_ostream &os, int indent) const override;

private:
  std::string block_sym_;
  IterTypeTag iter_type_;
  Expr trip_count_;
};

// =============================================================================
// ConstraintScope
// =============================================================================

/// Per-symbol metadata: type string and optional natural upper bound.
struct SymbolInfo {
  std::string type;        // always "int" for now
  int64_t natural_ub = -1; // -1 = unknown / not provided by loom.sym
  int64_t alignment = 1;   // hardware alignment factor for this symbol
};

/// ConstraintScope: Captures constraint metadata from a computation variant.
/// Contains symbolic block sizes, loop iteration counts, and assembly formulas.
struct ConstraintScope {
  // metadata.symbols: maps symbol name (e.g., "tile_m") to SymbolInfo
  std::map<std::string, SymbolInfo> symbols;
  // metadata.L1_footprint: symbolic @L1 allocation sizes by usage class plus
  // the capacity available to the solver-side memory model
  L1FootprintByScope l1_footprint;
  // metadata.datatype: element type shared by all @L1 allocations (e.g., "f32")
  std::string datatype;
  // metadata.iter_num.seq_iter: symbolic trip count of the sequential loop
  Expr seq_iter;
  // metadata.iter_num.temp_iter: symbolic trip counts of temporal loops
  std::vector<Expr> temp_iter;
  // hard_constraints: constraints that every valid block-size assignment must satisfy
  std::vector<ConstraintExpr> hard_constraints;
  // metadata.booleans: symbolic boolean variables to be optimized by the solver.
  // Represented as integer symbols constrained to {0, 1} in the SMT model.
  std::vector<std::string> booleans;

  /// Public API for appending a new hard constraint to the ETG.
  void pushHardConstraint(ConstraintExpr constraint);

  llvm::json::Value toJSON() const;
};

// =============================================================================
// VariantETG
// =============================================================================

/// VariantETG: Loop-structure-aware Execution Task Graph builder for a
/// computation variant. The kernel's compute and memory operations are
/// captured into a `KernelBlock`, with nested scf.for loops recursively
/// modelled as `for_loop_block` stages.
class VariantETG {
public:
  VariantETG(llvm::StringRef name, const HWOpRegistry *registry);

  llvm::StringRef getVariantName() const { return variant_name_; }
  const KernelBlock &getKernelBlock() const { return kernel_block_; }
  const ConstraintScope &getConstraintScope() const { return constraint_scope_; }

  /// Build the kernel's ETG by walking the function body, recursing into
  /// scf.for loops to produce nested for_loop_block stages and walking
  /// through affine.parallel loops transparently.
  void buildFromFunc(mlir::func::FuncOp func_op);

  /// Build constraint scope from a func operation.
  /// Extracts symbolic block sizes and global loop iteration counts.
  void buildConstraintScope(mlir::func::FuncOp func_op);

  /// Build and push all hard constraints through the centralized pipeline.
  void buildHardConstraints(mlir::func::FuncOp func_op);

  void dump(llvm::raw_ostream &os) const;
  llvm::json::Value toJSON() const;

private:
  std::string variant_name_;
  KernelBlock kernel_block_;
  ConstraintScope constraint_scope_;
  const HWOpRegistry *hw_registry_;

  // Recursive scope population. Walks `region`'s direct ops, dispatching
  // each into the given load / compute / store scope according to its kind.
  void populateScopesFromRegion(mlir::Region &region, Scope &load_scope,
                                Scope &compute_scope, Scope &store_scope);

  void dispatchToComputeQueues(mlir::Operation *op,
                               WorkloadStageBody &target);
  void dispatchNamedOp(mlir::Operation *op, WorkloadStageBody &target);
  void dispatchGenericOp(mlir::Operation *op, WorkloadStageBody &target);
  void dispatchToDataMoverQueues(mlir::Operation *op,
                                 WorkloadStageBody &target);

  void collectSymbols(mlir::func::FuncOp func_op);
  void analyzeLoopIterations(mlir::func::FuncOp func_op);
  void addIterDivisibilityConstraints(const Expr &iter);
};

} // namespace lcs
} // namespace loom

#endif // LOOM_LCS_STAGED_ETG_BUILDER_H
