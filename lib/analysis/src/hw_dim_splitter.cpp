#include "hw_dim_splitter.h"

#include <algorithm>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/StringMap.h"
#include <cassert>

namespace loom {

//===----------------------------------------------------------------------===//
// orderedFactorizations
//===----------------------------------------------------------------------===//

llvm::SmallVector<llvm::SmallVector<int64_t>>
HWDimSplitter::orderedFactorizations(int64_t n, unsigned k) {
  llvm::SmallVector<llvm::SmallVector<int64_t>> results;

  if (k == 1) {
    if (n > 1)
      results.push_back({n});
    return results;
  }

  // Try each divisor d of n where d > 1.
  for (int64_t d = 2; d <= n; ++d) {
    if (n % d != 0)
      continue;
    int64_t remainder = n / d;
    // Recurse to factor the remainder into (k-1) parts.
    auto subResults = orderedFactorizations(remainder, k - 1);
    for (auto &sub : subResults) {
      llvm::SmallVector<int64_t> combined;
      combined.push_back(d);
      combined.append(sub.begin(), sub.end());
      results.push_back(std::move(combined));
    }
  }

  return results;
}

//===----------------------------------------------------------------------===//
// enumeratePartitions
//===----------------------------------------------------------------------===//

void HWDimSplitter::enumeratePartitions(
    unsigned total, unsigned numParts,
    llvm::SmallVector<unsigned> &current,
    llvm::SmallVector<llvm::SmallVector<unsigned>> &results) {
  if (numParts == 1) {
    current.push_back(total);
    results.push_back(current);
    current.pop_back();
    return;
  }

  // Each part must be >= 1, so the remaining (numParts - 1) parts need
  // at least (numParts - 1).  Current part ranges from 1 to
  // total - (numParts - 1).
  unsigned maxVal = total - (numParts - 1);
  for (unsigned v = 1; v <= maxVal; ++v) {
    current.push_back(v);
    enumeratePartitions(total - v, numParts - 1, current, results);
    current.pop_back();
  }
}

//===----------------------------------------------------------------------===//
// cartesianFactorizations
//===----------------------------------------------------------------------===//

void HWDimSplitter::cartesianFactorizations(
    unsigned dimIdx,
    const llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<int64_t>>>
        &perDimFacts,
    llvm::SmallVector<llvm::SmallVector<int64_t>> &current,
    llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<int64_t>>>
        &results) {
  if (dimIdx == perDimFacts.size()) {
    results.push_back(current);
    return;
  }
  for (const auto &factorization : perDimFacts[dimIdx]) {
    current.push_back(factorization);
    cartesianFactorizations(dimIdx + 1, perDimFacts, current, results);
    current.pop_back();
  }
}

//===----------------------------------------------------------------------===//
// generateAllSplits
//===----------------------------------------------------------------------===//

llvm::SmallVector<HWDimSplit>
HWDimSplitter::generateAllSplits(
    unsigned P, const llvm::SmallVector<SpatialDimInfo> &hwDims) {
  const unsigned D = static_cast<unsigned>(hwDims.size());
  assert(P >= D && "P must be >= D (number of HW dims)");

  llvm::SmallVector<HWDimSplit> allSplits;

  // 1. Enumerate all partitions of P into D parts (each >= 1).
  //    partition[i] = number of logical dims from HW dim i.
  llvm::SmallVector<llvm::SmallVector<unsigned>> partitions;
  llvm::SmallVector<unsigned> partCurrent;
  enumeratePartitions(P, D, partCurrent, partitions);

  for (const auto &partition : partitions) {
    // 2. For each HW dim, compute all ordered factorizations with the
    //    required number of factors.
    llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<int64_t>>>
        perDimFacts;
    bool valid = true;
    for (unsigned i = 0; i < D; ++i) {
      int64_t dimSize = hwDims[i].size.value_or(1);
      auto facts = orderedFactorizations(dimSize, partition[i]);
      if (facts.empty()) {
        valid = false;
        break;
      }
      perDimFacts.push_back(std::move(facts));
    }
    if (!valid)
      continue;

    // 3. Cartesian product of per-dim factorizations.
    llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<int64_t>>>
        combinations;
    llvm::SmallVector<llvm::SmallVector<int64_t>> combCurrent;
    cartesianFactorizations(0, perDimFacts, combCurrent, combinations);

    // 4. For each combination, build the LogicalDim entries and sort them.
    for (const auto &combo : combinations) {
      // combo[i] = ordered factors for HW dim i, inner to outer.
      // Factor at index j has level j.
      llvm::SmallVector<LogicalDim> logicalDims;
      for (unsigned i = 0; i < D; ++i) {
        const auto &factors = combo[i];
        for (unsigned j = 0; j < factors.size(); ++j) {
          LogicalDim ld;
          ld.size = factors[j];
          ld.level = j;
          ld.sourceName = hwDims[i].symbolName;
          ld.sourceIdx = i;
          logicalDims.push_back(ld);
        }
      }

      // Sort by (level ASC, sourceIdx ASC).
      std::stable_sort(logicalDims.begin(), logicalDims.end(),
                       [](const LogicalDim &a, const LogicalDim &b) {
                         if (a.level != b.level)
                           return a.level < b.level;
                         return a.sourceIdx < b.sourceIdx;
                       });

      HWDimSplit split;
      split.logicalDims = std::move(logicalDims);
      allSplits.push_back(std::move(split));
    }
  }

  assert(!allSplits.empty() &&
         "No valid HW dim split found for the given P and hardware dims");
  return allSplits;
}

using namespace mlir;

mlir::Value
MeshCoordinateSystem::emitLinearIndex(mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      const AxisLinearIndex &axis) const {
  if (axis.ivs.empty()) {
    return builder.create<arith::ConstantIndexOp>(loc, 0);
  }
  Value result = axis.ivs[0];
  int64_t stride = axis.tileSizes[0];
  for (unsigned i = 1; i < axis.ivs.size(); ++i) {
    Value strideVal = builder.create<arith::ConstantIndexOp>(loc, stride);
    Value term = builder.create<arith::MulIOp>(loc, axis.ivs[i], strideVal);
    result = builder.create<arith::AddIOp>(loc, result, term);
    stride *= axis.tileSizes[i];
  }
  return result;
}

mlir::Value MeshCoordinateSystem::emitLinearIndexWithOverride(
    mlir::OpBuilder &builder, mlir::Location loc, const AxisLinearIndex &axis,
    unsigned overrideLevelIdx, int64_t overrideValue) const {
  if (axis.ivs.empty()) {
    return builder.create<arith::ConstantIndexOp>(loc, 0);
  }

  Value overrideIV =
      builder.create<arith::ConstantIndexOp>(loc, overrideValue);

  Value result = (overrideLevelIdx == 0) ? overrideIV : axis.ivs[0];
  int64_t stride = axis.tileSizes[0];
  for (unsigned i = 1; i < axis.ivs.size(); ++i) {
    Value iv = (i == overrideLevelIdx) ? overrideIV : axis.ivs[i];
    Value strideVal = builder.create<arith::ConstantIndexOp>(loc, stride);
    Value term = builder.create<arith::MulIOp>(loc, iv, strideVal);
    result = builder.create<arith::AddIOp>(loc, result, term);
    stride *= axis.tileSizes[i];
  }
  return result;
}

mlir::Value MeshCoordinateSystem::emitLinearIndexWithMultiOverride(
    mlir::OpBuilder &builder, mlir::Location loc, const AxisLinearIndex &axis,
    const llvm::DenseMap<unsigned, int64_t> &overrides) const {
  if (axis.ivs.empty()) {
    return arith::ConstantIndexOp::create(builder, loc, 0);
  }

  auto getIV = [&](unsigned i) -> Value {
    auto it = overrides.find(i);
    if (it != overrides.end())
      return arith::ConstantIndexOp::create(builder, loc, it->second);
    return axis.ivs[i];
  };

  Value result = getIV(0);
  int64_t stride = axis.tileSizes[0];
  for (unsigned i = 1; i < axis.ivs.size(); ++i) {
    Value iv = getIV(i);
    Value strideVal = arith::ConstantIndexOp::create(builder, loc, stride);
    Value term = arith::MulIOp::create(builder, loc, iv, strideVal);
    result = arith::AddIOp::create(builder, loc, result, term);
    stride *= axis.tileSizes[i];
  }
  return result;
}

MeshCoordinateSystem
MeshCoordinateSystem::fromEnclosingLoops(
    mlir::Operation *op, llvm::ArrayRef<std::string> meshDimNames) {
  // Collect spatial parallel loops with their dim/level metadata.
  struct LoopEntry {
    affine::AffineParallelOp parOp;
    std::string dimName;
    int64_t level;
    int64_t upperBound;
  };

  llvm::StringMap<llvm::SmallVector<LoopEntry>> dimLoops;

  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto par = dyn_cast<affine::AffineParallelOp>(parent);
    if (!par)
      continue;

    auto dimAttr = par->getAttrOfType<SymbolRefAttr>("loom.physical_dim");
    auto levelAttr = par->getAttrOfType<IntegerAttr>("loom.logical_level");
    if (!dimAttr || !levelAttr)
      continue;

    // Extract constant upper bound.
    AffineMap ubMap = par.getUpperBoundsMap();
    assert(ubMap.getNumResults() == 1 && "spatial loop must have single UB");
    auto constExpr = dyn_cast<AffineConstantExpr>(ubMap.getResult(0));
    assert(constExpr && "spatial loop must have constant UB");
    int64_t ub = constExpr.getValue();

    std::string dimName =
        dimAttr.getRootReference().getValue().str();
    int64_t level = levelAttr.getInt();

    dimLoops[dimName].push_back({par, dimName, level, ub});
  }

  // Sort each dim's entries by level ascending (innermost first).
  for (auto &entry : dimLoops) {
    llvm::sort(entry.second,
               [](const LoopEntry &a, const LoopEntry &b) {
                 return a.level < b.level;
               });
  }

  // Build the MeshCoordinateSystem.
  MeshCoordinateSystem meshCoords;

  for (unsigned dimIdx = 0; dimIdx < meshDimNames.size() && dimIdx < 2;
       ++dimIdx) {
    AxisLinearIndex &axis =
        (dimIdx == 0) ? meshCoords.xAxis : meshCoords.yAxis;
    axis.sourceIdx = dimIdx;

    auto it = dimLoops.find(meshDimNames[dimIdx]);
    if (it == dimLoops.end())
      continue;

    unsigned logicalIdx = 0;
    for (auto &entry : it->second) {
      // Each spatial parallel loop has exactly one IV (block argument).
      assert(entry.parOp.getBody()->getNumArguments() == 1 &&
             "spatial parallel loop must have exactly one IV");
      axis.ivs.push_back(entry.parOp.getBody()->getArgument(0));
      axis.tileSizes.push_back(entry.upperBound);
      axis.logicalDimIndices.push_back(logicalIdx++);
    }
  }

  return meshCoords;
}

} // namespace loom
