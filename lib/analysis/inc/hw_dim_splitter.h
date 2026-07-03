#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace loom {

/// Describes how a single physical mesh axis (x or y) was split into
/// multiple parallel loop levels after hardware mapping.
struct AxisLinearIndex {
  unsigned sourceIdx;                          // 0 = x, 1 = y
  llvm::SmallVector<mlir::Value> ivs;          // IVs ordered innermost to outermost
  llvm::SmallVector<int64_t> tileSizes;        // Tile size at each level
  llvm::SmallVector<unsigned> logicalDimIndices; // Index into logicalDims for each level
};

/// Complete 2D mesh coordinate system after hardware mapping.
struct MeshCoordinateSystem {
  AxisLinearIndex xAxis; // sourceIdx = 0
  AxisLinearIndex yAxis; // sourceIdx = 1

  /// Emit the linear index SSA value for the given axis using all real IVs.
  mlir::Value emitLinearIndex(mlir::OpBuilder &builder, mlir::Location loc,
                              const AxisLinearIndex &axis) const;

  /// Emit the linear index with a specific level's IV replaced by a constant.
  mlir::Value emitLinearIndexWithOverride(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          const AxisLinearIndex &axis,
                                          unsigned overrideLevelIdx,
                                          int64_t overrideValue) const;

  /// Emit the linear index with multiple levels' IVs replaced by constants.
  /// overrides maps levelIdx -> overrideValue.
  mlir::Value emitLinearIndexWithMultiOverride(
      mlir::OpBuilder &builder, mlir::Location loc,
      const AxisLinearIndex &axis,
      const llvm::DenseMap<unsigned, int64_t> &overrides) const;

  /// Reconstruct a MeshCoordinateSystem from the loop attributes on the IR.
  /// Walks up the parent chain from `op`, collecting affine.parallel loops
  /// with loom.physical_dim and loom.logical_level attributes.
  /// meshDimNames provides the ordering: meshDimNames[0] -> xAxis, [1] -> yAxis.
  static MeshCoordinateSystem
  fromEnclosingLoops(mlir::Operation *op,
                     llvm::ArrayRef<std::string> meshDimNames);
};

} // namespace loom
