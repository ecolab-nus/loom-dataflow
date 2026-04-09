#include "mapping_prioritizer.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/Support/raw_ostream.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#include <algorithm>
#include <cassert>
#include <numeric>

using namespace mlir;

namespace loom {

//===----------------------------------------------------------------------===//
// collectParallelIterDeps
//===----------------------------------------------------------------------===//

void MappingPrioritizer::collectParallelIterDeps(
    Value v, const SmallPtrSetImpl<Value> &parallelIVs,
    SmallPtrSetImpl<Value> &deps, SmallPtrSetImpl<Value> &visited) {
  // If this value is one of the root parallel IVs, record it.
  if (parallelIVs.count(v)) {
    deps.insert(v);
    return;
  }
  // Memoisation: don't revisit.
  if (!visited.insert(v).second)
    return;
  // Block argument (e.g. inner affine.for IV) with no defining op — stop.
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return;
  // Recurse into all operands of the defining op.
  for (Value operand : defOp->getOperands())
    collectParallelIterDeps(operand, parallelIVs, deps, visited);
}

//===----------------------------------------------------------------------===//
// computeIterWeights
//===----------------------------------------------------------------------===//

llvm::SmallVector<int64_t>
MappingPrioritizer::computeIterWeights(
    func::FuncOp func, affine::AffineParallelOp rootParallel) {
  const unsigned P = rootParallel.getNumDims();
  llvm::SmallVector<int64_t> weights(P, 0);

  // Collect the root parallel IVs (block arguments of its body).
  SmallPtrSet<Value, 8> parallelIVs;
  for (Value iv : rootParallel.getBody()->getArguments())
    parallelIVs.insert(iv);

  func.walk([&](loom::SubviewOp subview) {
    // Compute the element count of the source memref.
    // All dimensions must be static at this pipeline stage.
    MemRefType srcType =
        llvm::cast<MemRefType>(subview.getSource().getType());
    int64_t size = 1;
    for (int64_t dim : srcType.getShape()) {
      assert(!ShapedType::isDynamic(dim) &&
             "SubviewOp source memref must have a static shape");
      size *= dim;
    }

    // Find which root parallel IVs contribute to any offset operand.
    SmallPtrSet<Value, 4> deps;
    SmallPtrSet<Value, 16> visited;
    for (OpFoldResult mixed : subview.getMixedOffsets()) {
      // Only dynamic offsets (Value) can depend on parallel IVs.
      if (Value v = dyn_cast<Value>(mixed))
        collectParallelIterDeps(v, parallelIVs, deps, visited);
    }

    // Accumulate size into the weight of each participating IV.
    for (Value dep : deps) {
      unsigned argNum = llvm::cast<BlockArgument>(dep).getArgNumber();
      weights[argNum] += size;
    }
  });

  return weights;
}

//===----------------------------------------------------------------------===//
// cartesianPerms
//===----------------------------------------------------------------------===//

void MappingPrioritizer::cartesianPerms(
    unsigned groupIdx,
    const llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<unsigned>>>
        &groupPerms,
    llvm::SmallVector<unsigned> &current,
    llvm::SmallVector<llvm::SmallVector<unsigned>> &results) {
  if (groupIdx == groupPerms.size()) {
    results.push_back(current);
    return;
  }
  // groupPerms[groupIdx] is the list of permutations for this group.
  // Each perm is a SmallVector<unsigned> of iter indices.
  for (const auto &perm : groupPerms[groupIdx]) {
    size_t prevSize = current.size();
    current.append(perm.begin(), perm.end());
    cartesianPerms(groupIdx + 1, groupPerms, current, results);
    current.resize(prevSize);
  }
}

//===----------------------------------------------------------------------===//
// generateBijectiveMappings
//===----------------------------------------------------------------------===//

llvm::SmallVector<DimBuckets>
MappingPrioritizer::generateBijectiveMappings(
    const llvm::SmallVector<int64_t> &weights, unsigned numHWDims) {
  assert(weights.size() == numHWDims &&
         "number of parallel iters must equal number of HW dims (P == D)");

  const unsigned P = static_cast<unsigned>(weights.size());

  // Build (weight, iterIdx) pairs and sort ascending by weight.
  // Stable sort so that iters with equal weight retain their original order
  // as the canonical starting permutation.
  llvm::SmallVector<std::pair<int64_t, unsigned>> pairs;
  pairs.reserve(P);
  for (unsigned i = 0; i < P; ++i)
    pairs.push_back({weights[i], i});
  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const auto &a, const auto &b) {
                     return a.first < b.first; // ascending weight
                   });

  // Group consecutive equal-weight pairs.
  // Each group is assigned a contiguous block of HW dim indices (implicit in
  // the Cartesian product position).  Within a group all permutations of iter
  // indices are emitted.
  // groupPerms[i] = list of permutations for group i;
  // groupPerms[i][j] = one permutation = SmallVector<unsigned> of iter indices.
  llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<unsigned>>> groupPerms;

  unsigned i = 0;
  while (i < P) {
    // Find the end of this equal-weight run.
    unsigned j = i + 1;
    while (j < P && pairs[j].first == pairs[i].first)
      ++j;

    // Collect iter indices for this group.
    llvm::SmallVector<unsigned> groupIters;
    for (unsigned k = i; k < j; ++k)
      groupIters.push_back(pairs[k].second);
    std::sort(groupIters.begin(), groupIters.end()); // canonical start

    // Enumerate all permutations of this group.
    llvm::SmallVector<llvm::SmallVector<unsigned>> permsForGroup;
    do {
      permsForGroup.push_back(groupIters);
    } while (std::next_permutation(groupIters.begin(), groupIters.end()));

    groupPerms.push_back(std::move(permsForGroup));
    i = j;
  }

  // Cartesian product of per-group permutations → combined orderings.
  // combined[j] = iter index that maps to HW dim j.
  llvm::SmallVector<llvm::SmallVector<unsigned>> combinedOrderings;
  llvm::SmallVector<unsigned> current;
  cartesianPerms(0, groupPerms, current, combinedOrderings);

  // Convert each combined ordering into a bijective DimBuckets.
  // mapping[iterIdx] = {hwDimIdx}
  llvm::SmallVector<DimBuckets> result;
  result.reserve(combinedOrderings.size());
  for (const auto &combined : combinedOrderings) {
    DimBuckets mapping(P);
    for (unsigned hwDim = 0; hwDim < P; ++hwDim)
      mapping[combined[hwDim]].push_back(hwDim);
    result.push_back(std::move(mapping));
  }
  return result;
}

//===----------------------------------------------------------------------===//
// generateBijectiveMappingsWithForcedFirst
//===----------------------------------------------------------------------===//

llvm::SmallVector<DimBuckets>
MappingPrioritizer::generateBijectiveMappingsWithForcedFirst(
    const llvm::SmallVector<int64_t> &weights, unsigned numHWDims,
    unsigned forcedIterIdx) {
  assert(weights.size() == numHWDims &&
         "number of parallel iters must equal number of HW dims (P == D)");
  assert(forcedIterIdx < weights.size());

  const unsigned P = static_cast<unsigned>(weights.size());

  // 1. Forced iter at HW dim 0.
  // 2. Build (weight, iterIdx) pairs for REMAINING iters.
  llvm::SmallVector<std::pair<int64_t, unsigned>> pairs;
  pairs.reserve(P - 1);
  for (unsigned i = 0; i < P; ++i) {
    if (i == forcedIterIdx)
      continue;
    pairs.push_back({weights[i], i});
  }
  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const auto &a, const auto &b) {
                     return a.first < b.first; // ascending weight
                   });

  // 3. Group by equal weight and get permutations.
  llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<unsigned>>> groupPerms;
  for (unsigned i = 0; i < pairs.size();) {
    unsigned j = i + 1;
    while (j < pairs.size() && pairs[j].first == pairs[i].first)
      ++j;

    llvm::SmallVector<unsigned> groupIters;
    for (unsigned k = i; k < j; ++k)
      groupIters.push_back(pairs[k].second);

    llvm::SmallVector<llvm::SmallVector<unsigned>> perms;
    std::sort(groupIters.begin(), groupIters.end());
    do {
      perms.push_back(groupIters);
    } while (std::next_permutation(groupIters.begin(), groupIters.end()));

    groupPerms.push_back(std::move(perms));
    i = j;
  }

  // 4. Combine: forcedIterIdx at 0, followed by all Cartesian permutations of
  // others.
  llvm::SmallVector<llvm::SmallVector<unsigned>> combinations;
  llvm::SmallVector<unsigned> start = {forcedIterIdx};
  if (P > 1) {
    cartesianPerms(0, groupPerms, start, combinations);
  } else {
    combinations.push_back(start);
  }

  // 5. Convert to DimBuckets.
  llvm::SmallVector<DimBuckets> result;
  for (const auto &comb : combinations) {
    DimBuckets buckets(P);
    for (unsigned hwDimIdx = 0; hwDimIdx < P; ++hwDimIdx) {
      unsigned iterIdx = comb[hwDimIdx];
      buckets[iterIdx].push_back(hwDimIdx);
    }
    result.push_back(std::move(buckets));
  }

  return result;
}

} // namespace loom
