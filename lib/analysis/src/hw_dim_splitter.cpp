#include "hw_dim_splitter.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/StringMap.h"
#include <cassert>

namespace loom {

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
