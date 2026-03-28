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
 */
mlir::OwningOpRef<mlir::ModuleOp>
EnumerateSpatialMappings(mlir::ModuleOp affineModule,
                         const HardwareInfo &hardwareInfo);
} // namespace loom
