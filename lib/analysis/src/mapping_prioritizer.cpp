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
#include "LoomInterfaces.h.inc"
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

bool dominates(const AxisScores &scores, unsigned lhs, unsigned rhs) {
  bool noWorse =
      scores.parallelism[lhs] >= scores.parallelism[rhs] &&
      scores.reuseVolume[lhs] >= scores.reuseVolume[rhs] &&
      scores.reuseAccessCount[lhs] >= scores.reuseAccessCount[rhs];
  bool strictlyBetter =
      scores.parallelism[lhs] > scores.parallelism[rhs] ||
      scores.reuseVolume[lhs] > scores.reuseVolume[rhs] ||
      scores.reuseAccessCount[lhs] > scores.reuseAccessCount[rhs];
  return noWorse && strictlyBetter;
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
  scores.reuseVolume.assign(P, 0);
  scores.reuseAccessCount.assign(P, 0);
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
      if (!deps.count(iv)) {
        scores.reuseVolume[i] += size;
        scores.reuseAccessCount[i] += 1;
      }
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
    const AxisScores &scores, std::optional<unsigned> reductionIdx) {
  assert(scores.reuseVolume.size() == scores.parallelism.size());
  assert(scores.reuseAccessCount.size() == scores.parallelism.size());
  assert(!scores.reuseVolume.empty());
  const unsigned P = static_cast<unsigned>(scores.reuseVolume.size());
  if (reductionIdx)
    assert(*reductionIdx < P);

  llvm::SmallVector<unsigned> remaining;
  remaining.reserve(reductionIdx ? P - 1 : P);
  for (unsigned i = 0; i < P; ++i) {
    if (reductionIdx && i == *reductionIdx)
      continue;
    remaining.push_back(i);
  }

  llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<unsigned>>> groupPerms;
  while (!remaining.empty()) {
    llvm::SmallVector<unsigned> front;
    llvm::SmallVector<unsigned> nextRemaining;
    for (unsigned candidate : remaining) {
      bool isDominated = false;
      for (unsigned other : remaining) {
        if (candidate == other)
          continue;
        if (dominates(scores, other, candidate)) {
          isDominated = true;
          break;
        }
      }
      if (isDominated)
        nextRemaining.push_back(candidate);
      else
        front.push_back(candidate);
    }

    llvm::SmallVector<llvm::SmallVector<unsigned>> perms;
    std::sort(front.begin(), front.end());
    do {
      perms.push_back(front);
    } while (std::next_permutation(front.begin(), front.end()));

    groupPerms.push_back(std::move(perms));
    remaining = std::move(nextRemaining);
  }

  llvm::SmallVector<llvm::SmallVector<unsigned>> suffixOrderings;
  llvm::SmallVector<unsigned> current;
  if (groupPerms.empty())
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
llvm::SmallVector<DimBuckets>
MappingPrioritizer::generateLevel0PairClaimMappings(
    const AxisScores &scores, llvm::ArrayRef<unsigned> level0DimIndices,
    llvm::ArrayRef<unsigned> nonLevel0DimIndices,
    std::optional<unsigned> reductionIdx) {
  assert(scores.reuseVolume.size() == scores.parallelism.size());
  assert(scores.reuseAccessCount.size() == scores.parallelism.size());
  assert(!scores.reuseVolume.empty());
  assert(level0DimIndices.size() == 2);
  assert(nonLevel0DimIndices.size() == scores.reuseVolume.size() - 1);

  const unsigned P = static_cast<unsigned>(scores.reuseVolume.size());
  auto orderings = enumeratePriorityOrderings(scores, reductionIdx);
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
