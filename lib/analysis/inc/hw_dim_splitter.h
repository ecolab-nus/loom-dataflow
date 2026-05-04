#pragma once

#include "hardware_info.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace loom {

/// A logical dimension produced by splitting a physical HW dim.
struct LogicalDim {
  int64_t size;            // Size of this logical dimension (e.g., 4, 2, 8)
  unsigned level;          // 0 = innermost; incremented per split level
  std::string sourceName;  // Original HW dim symbol (e.g., "dim_x")
  unsigned sourceIdx;      // Index of source in spatialDimInfoVec
};

/// A complete split configuration producing exactly P logical dims.
struct HWDimSplit {
  llvm::SmallVector<LogicalDim> logicalDims; // Ordered: (level ASC, sourceIdx ASC)
};

/// Generates all ways to split D physical HW dims into P logical dims.
///
/// Given P parallel iterators and D physical hardware dimensions, this class
/// enumerates all valid ways to split the HW dims so that exactly P logical
/// dims are produced (P >= D).  Each HW dim contributes at least one logical
/// dim.  Factors of 1 are excluded by default, but can be enabled for
/// callers that model degenerate logical dimensions explicitly.
///
/// Example: P=3, D=2, dim_x=8, dim_y=8
///   Partitions: (p_x=2, p_y=1) and (p_x=1, p_y=2)
///   For p_x=2: x factored into (2,4) or (4,2)
///   Total splits: 4
class HWDimSplitter {
public:
  /// Produce all valid HWDimSplit configurations.
  /// Requires P >= D (each HW dim contributes >= 1 logical dim).
  /// Asserts if no valid split exists.
  llvm::SmallVector<HWDimSplit>
  generateAllSplits(unsigned P,
                    const llvm::SmallVector<SpatialDimInfo> &hwDims,
                    bool allowSizeOne = false);

private:
  /// All ordered k-factorizations of n. Each factor is > 1 unless
  /// allowSizeOne is set.
  /// E.g., orderedFactorizations(8, 2) = {{2,4}, {4,2}}
  llvm::SmallVector<llvm::SmallVector<int64_t>>
  orderedFactorizations(int64_t n, unsigned k, bool allowSizeOne);

  /// Enumerate all integer partitions of total into numParts parts, each >= 1.
  void enumeratePartitions(
      unsigned total, unsigned numParts,
      llvm::SmallVector<unsigned> &current,
      llvm::SmallVector<llvm::SmallVector<unsigned>> &results);

  /// Cartesian product helper for combining per-dim factorizations.
  void cartesianFactorizations(
      unsigned dimIdx,
      const llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<int64_t>>>
          &perDimFacts,
      llvm::SmallVector<llvm::SmallVector<int64_t>> &current,
      llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<int64_t>>>
          &results);
};

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
