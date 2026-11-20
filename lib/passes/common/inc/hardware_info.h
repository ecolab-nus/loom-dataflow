#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>

namespace loom {

/**
 * \brief Hardware spatial dimension description parsed from the ADL module.
 */
struct SpatialDimInfo {
  std::string name;
  std::optional<int64_t> size;
  std::string symbolName; // Symbol name for SymbolRefAttr references
  unsigned level = 0;     // Logical split level (0 = innermost)
  unsigned sourceIdx = 0; // Original HW dim index
};

/**
 * \brief Mapping of parallel iteration dimension to hardware spatial dimension.
 * Used to track core utilization constraints.
 */
struct ParallelToHWMapping {
  unsigned parallelIterIdx; // 0=M, 1=N (from affine.parallel order)
  unsigned hwDimIdx;        // Index into spatialDimInfoVec (0=x, 1=y)
  int64_t hwDimSize;        // Size of the hardware dimension (e.g., 8)
};

struct HardwareInfo {
  llvm::SmallVector<SpatialDimInfo> spatialDimInfoVec;
};

typedef llvm::SmallVector<llvm::SmallVector<unsigned>> DimBuckets;

/**
 * \brief Collect hardware info from an ADL module.
 *
 * Finds the adl.arch.scale op and extracts its spatial dimension operands.
 * Asserts exactly one ArchScaleOp exists in the module.
 */
mlir::LogicalResult GetHardwareInfoForExploration(mlir::ModuleOp hwModule,
                                                  HardwareInfo &hardwareInfo);

/**
 * \brief Enumerate all unique mappings and emit one function clone per mapping.
 *
 * When \p fullOccupancy is true, use the physical hardware dimensions exactly
 * as declared. When false (the default), also enumerate partial occupancy
 * variants for every hardware dimension.
 */
///
/// When a parallel dimension has more iterations than the cores mapped to it,
/// the remainder becomes a wave loop. \p blockedWaves selects how iterations
/// are distributed: false (cyclic, the default) gives core c the iterations
/// c, c + C, c + 2C, ...; true (blocked) gives it the contiguous range
/// [c * W, (c + 1) * W) with W the wave count, so consecutive waves on one core
/// visit consecutive iterations.
mlir::OwningOpRef<mlir::ModuleOp>
EnumerateSpatialMappings(mlir::ModuleOp affineModule,
                         const HardwareInfo &hardwareInfo,
                         bool fullOccupancy = false,
                         bool blockedWaves = false);

/**
 * \brief Add function variants that run a different set of loops spatially.
 *
 * For each function whose outermost affine.parallel directly contains a
 * perfectly nested chain of independent scf.for loops, clone the function once
 * per pruned spatial set: chain loops are promoted into the parallel loop and
 * parallel dimensions demoted to sequential loops. Clones are inserted after
 * the original, which stays unchanged.
 */
void ExploreParallelSets(mlir::ModuleOp module,
                         const HardwareInfo &hardwareInfo);

/**
 * \brief Add function variants that split loops at floor-division boundaries.
 *
 * For each loop of a function's promotable scf.for chain whose index is used
 * as `index / K` (K a constant proper divisor of its static trip count), clone
 * the function with the loop split into a loop over the trip/K groups and a
 * loop over the K members (index = g * K + i, index / K = g, index % K = i).
 * Loads indexed by the group then no longer depend on the member loop, so
 * reuse analysis can share them. The clones are marked for
 * ExploreParallelSets, which keeps only their variants that run a split loop
 * spatially and then erases them; run it before ExploreParallelSets.
 */
void SplitFloorDivLoops(mlir::ModuleOp module);
} // namespace loom
