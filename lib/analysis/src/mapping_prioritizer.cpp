#include "mapping_prioritizer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/Support/raw_ostream.h"

#include "LoomDialect.h.inc"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#include <algorithm>
#include <cassert>
#include <numeric>

using namespace mlir;

namespace loom {

namespace {

std::optional<int64_t> getStaticUpperBound(Value v) {
  if (!v)
    return std::nullopt;

  APInt constValue;
  if (matchPattern(v, m_ConstantInt(&constValue)))
    return constValue.getSExtValue();

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  if (defOp->getName().getStringRef() == "loom.sym") {
    if (auto ub = defOp->getAttrOfType<IntegerAttr>("upper_bound"))
      return ub.getInt();
    return std::nullopt;
  }

  if (auto ceilDiv = dyn_cast<arith::CeilDivUIOp>(defOp))
    return getStaticUpperBound(ceilDiv.getLhs());

  if (auto mul = dyn_cast<arith::MulIOp>(defOp)) {
    if (auto lhsConst = getStaticUpperBound(mul.getLhs())) {
      APInt rhsConst;
      if (matchPattern(mul.getRhs(), m_ConstantInt(&rhsConst)))
        return *lhsConst * rhsConst.getSExtValue();
    }
    if (auto rhsConst = getStaticUpperBound(mul.getRhs())) {
      APInt lhsConst;
      if (matchPattern(mul.getLhs(), m_ConstantInt(&lhsConst)))
        return lhsConst.getSExtValue() * *rhsConst;
    }
  }

  return std::nullopt;
}

int64_t getStaticIterUpperBound(affine::AffineParallelOp par,
                                unsigned iterIdx) {
  AffineMap ubMap = par.getUpperBoundMap(iterIdx);
  if (ubMap.getNumResults() != 1)
    return 1;

  AffineExpr expr = ubMap.getResult(0);
  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr))
    return constExpr.getValue();

  if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    unsigned operandIdx = ubMap.getNumDims() + symExpr.getPosition();
    ValueRange operands = par.getUpperBoundsOperands();
    if (operandIdx < operands.size()) {
      if (auto ub = getStaticUpperBound(operands[operandIdx]))
        return *ub;
    }
  }

  par.emitWarning() << "could not compute static upper bound for iter "
                    << iterIdx << "; using 1";
  return 1;
}

struct AxisPriority {
  int64_t primary;
  int64_t secondary;

  bool operator==(const AxisPriority &other) const {
    return primary == other.primary && secondary == other.secondary;
  }
};

AxisPriority getPriority(const AxisScores &scores, PriorityKey key,
                         unsigned iterIdx) {
  switch (key) {
  case PriorityKey::Reuse:
    return {scores.reuse[iterIdx], 0};
  case PriorityKey::ParallelismThenReuse:
    return {scores.parallelism[iterIdx], scores.reuse[iterIdx]};
  }
  llvm_unreachable("unknown priority key");
}

} // namespace

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
// computeAxisScores
//===----------------------------------------------------------------------===//

AxisScores
MappingPrioritizer::computeAxisScores(
    func::FuncOp func, affine::AffineParallelOp rootParallel) {
  const unsigned P = rootParallel.getNumDims();
  AxisScores scores;
  scores.reuse.assign(P, 0);
  scores.parallelism.reserve(P);
  for (unsigned i = 0; i < P; ++i)
    scores.parallelism.push_back(getStaticIterUpperBound(rootParallel, i));

  // Collect the root parallel IVs (block arguments of its body).
  SmallPtrSet<Value, 8> parallelIVs;
  for (unsigned i = 0; i < P; ++i)
    parallelIVs.insert(rootParallel.getBody()->getArgument(i));

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

    // Reuse is data whose offsets do not depend on this axis.
    for (unsigned i = 0; i < P; ++i) {
      Value iv = rootParallel.getBody()->getArgument(i);
      if (!deps.count(iv))
        scores.reuse[i] += size;
    }
  });

  return scores;
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
// enumeratePriorityOrderings
//===----------------------------------------------------------------------===//

llvm::SmallVector<llvm::SmallVector<unsigned>>
MappingPrioritizer::enumeratePriorityOrderings(
    const AxisScores &scores, PriorityKey key,
    std::optional<unsigned> reductionIdx) {
  assert(scores.reuse.size() == scores.parallelism.size());
  assert(!scores.reuse.empty());
  const unsigned P = static_cast<unsigned>(scores.reuse.size());
  if (reductionIdx)
    assert(*reductionIdx < P);

  llvm::SmallVector<std::pair<AxisPriority, unsigned>> pairs;
  pairs.reserve(reductionIdx ? P - 1 : P);
  for (unsigned i = 0; i < P; ++i) {
    if (reductionIdx && i == *reductionIdx)
      continue;
    pairs.push_back({getPriority(scores, key, i), i});
  }

  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const auto &a, const auto &b) {
                     if (a.first.primary != b.first.primary)
                       return a.first.primary > b.first.primary;
                     return a.first.secondary > b.first.secondary;
                   });

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

  llvm::SmallVector<llvm::SmallVector<unsigned>> suffixOrderings;
  llvm::SmallVector<unsigned> current;
  if (pairs.empty())
    suffixOrderings.push_back({});
  else
    cartesianPerms(0, groupPerms, current, suffixOrderings);

  llvm::SmallVector<llvm::SmallVector<unsigned>> orderings;
  orderings.reserve(suffixOrderings.size());
  for (auto ordering : suffixOrderings) {
    if (reductionIdx)
      ordering.insert(ordering.begin(), *reductionIdx);
    orderings.push_back(std::move(ordering));
  }
  return orderings;
}

//===----------------------------------------------------------------------===//
// generateBijectiveMappings
//===----------------------------------------------------------------------===//

llvm::SmallVector<DimBuckets>
MappingPrioritizer::generateBijectiveMappings(
    const AxisScores &scores, unsigned numHWDims,
    std::optional<unsigned> reductionIdx) {
  assert(scores.reuse.size() == numHWDims &&
         scores.parallelism.size() == numHWDims &&
         "number of parallel iters must equal number of HW dims (P == D)");

  auto orderings =
      enumeratePriorityOrderings(scores, PriorityKey::Reuse, reductionIdx);

  // Convert each combined ordering into a bijective DimBuckets.
  // mapping[iterIdx] = {hwDimIdx}
  llvm::SmallVector<DimBuckets> result;
  result.reserve(orderings.size());
  for (const auto &priority : orderings) {
    DimBuckets mapping(numHWDims);
    for (unsigned hwDim = 0; hwDim < numHWDims; ++hwDim)
      mapping[priority[hwDim]].push_back(hwDim);
    result.push_back(std::move(mapping));
  }
  return result;
}

llvm::SmallVector<DimBuckets>
MappingPrioritizer::generateDoubledLevel0Mappings(
    const AxisScores &scores, llvm::ArrayRef<unsigned> level0DimIndices,
    llvm::ArrayRef<unsigned> nonLevel0DimIndices,
    std::optional<unsigned> reductionIdx) {
  assert(scores.reuse.size() == scores.parallelism.size());
  assert(!scores.reuse.empty());
  assert(level0DimIndices.size() == 2);
  assert(nonLevel0DimIndices.size() == scores.reuse.size() - 1);

  const unsigned P = static_cast<unsigned>(scores.reuse.size());
  auto orderings = enumeratePriorityOrderings(
      scores, PriorityKey::ParallelismThenReuse, reductionIdx);
  llvm::SmallVector<DimBuckets> result;
  result.reserve(orderings.size());

  for (const auto &priority : orderings) {
    DimBuckets buckets(P);
    buckets[priority[0]].push_back(level0DimIndices[0]);
    buckets[priority[0]].push_back(level0DimIndices[1]);
    for (unsigned idx = 1; idx < P; ++idx)
      buckets[priority[idx]].push_back(nonLevel0DimIndices[idx - 1]);
    result.push_back(std::move(buckets));
  }

  return result;
}

} // namespace loom
