#pragma once

#include "hardware_info.h" // DimBuckets typedef
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace loom {

/// Per-axis scores used to derive priority orderings.
struct AxisScores {
  /// Sum of source memref sizes for loom.subview ops whose offsets do not
  /// depend on axis i. Higher means more data can be shared along that axis.
  llvm::SmallVector<int64_t> reuse;
  /// Static upper bound on iter i's iteration range.
  llvm::SmallVector<int64_t> parallelism;
};

/// Priority-based mapping selector.
///
/// Replaces the combinatorial MappingEnumerator. Given a function with an
/// outermost affine.parallel, this class:
///   1. Computes reuse and parallelism scores for each parallel iterator.
///   2. Produces DimBuckets orderings from a partial order over those scores.
class MappingPrioritizer {
public:
  /// Compute both axis scores in one walk.
  AxisScores computeAxisScores(mlir::func::FuncOp func,
                               mlir::affine::AffineParallelOp rootParallel);

  /// Enumerate every linearisation of the partial order induced by
  /// (parallelism, reuse). With reductionIdx set, that iter is pinned at
  /// position 0.
  llvm::SmallVector<llvm::SmallVector<unsigned>>
  enumeratePriorityOrderings(const AxisScores &scores,
                             std::optional<unsigned> reductionIdx);

  /// Generate mappings where the highest-priority iter claims the two level-0
  /// logical dims. Priority order is sorted by parallelism DESC, reuse DESC.
  llvm::SmallVector<DimBuckets> generateLevel0PairClaimMappings(
      const AxisScores &scores, llvm::ArrayRef<unsigned> level0DimIndices,
      llvm::ArrayRef<unsigned> nonLevel0DimIndices,
      std::optional<unsigned> reductionIdx);

private:
  /// Recursively collect all root parallel IV block args that `v`
  /// transitively depends on. Uses `visited` to avoid redundant traversal.
  void collectParallelIterDeps(
      mlir::Value v,
      const llvm::SmallPtrSetImpl<mlir::Value> &parallelIVs,
      llvm::SmallPtrSetImpl<mlir::Value> &deps,
      llvm::SmallPtrSetImpl<mlir::Value> &visited);

  /// Cartesian product of per-group permutations.
  /// groupPerms[i] is the list of permutations for group i.
  /// current accumulates the combined ordering; results collects final ones.
  // groupPerms[i] = list of permutations for group i;
  // groupPerms[i][j] = one permutation = SmallVector<unsigned> of iter indices.
  void cartesianPerms(
      unsigned groupIdx,
      const llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<unsigned>>>
          &groupPerms,
      llvm::SmallVector<unsigned> &current,
      llvm::SmallVector<llvm::SmallVector<unsigned>> &results);

};

} // namespace loom
