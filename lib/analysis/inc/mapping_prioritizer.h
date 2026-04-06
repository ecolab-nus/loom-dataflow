#pragma once

#include "hardware_info.h" // DimBuckets typedef
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace loom {

/// Priority-based bijective mapping selector.
///
/// Replaces the combinatorial MappingEnumerator. Given a function with an
/// outermost affine.parallel, this class:
///   1. Computes a dependency weight for each parallel iterator by analysing
///      which loom.subview ops depend on that iterator (via offset SSA chains)
///      and accumulating the source memref sizes.
///   2. Produces bijective DimBuckets orderings where parallel iterators are
///      assigned to hardware dims in ascending weight order (lower weight →
///      lower HW dim index). Ties enumerate all permutations of tied iters.
class MappingPrioritizer {
public:
  /// Compute dep_weight[i] for each iter arg of rootParallel.
  ///   dep_weight(p) = Σ size(op) for loom.subview ops where
  ///                   p appears in any offset operand's def-use chain.
  /// size(op) = product of static dimensions of the subview source memref.
  /// All source memref dimensions must be static (asserted).
  /// Returns a vector of length rootParallel.getNumDims().
  llvm::SmallVector<int64_t>
  computeIterWeights(mlir::func::FuncOp func,
                     mlir::affine::AffineParallelOp rootParallel);

  /// Given per-iter weights and numHWDims (must equal weights.size()),
  /// produce all bijective DimBuckets orderings:
  ///   - Sort iter indices by ascending weight (lower weight → HW dim 0).
  ///   - Group consecutive iters with equal weight.
  ///   - Enumerate all permutations within each tied group.
  ///   - Take Cartesian product across groups.
  ///   - mapping[iter_i] = {hw_dim_j}  (exactly one HW dim per iter).
  /// Asserts weights.size() == numHWDims.
  llvm::SmallVector<DimBuckets>
  generateBijectiveMappings(const llvm::SmallVector<int64_t> &weights,
                            unsigned numHWDims);

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
