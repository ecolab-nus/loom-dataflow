#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>

namespace loom {

/**
 * \brief Hardware spatial dimension description parsed from the DF module.
 */
struct SpatialDimInfo {
  std::string name;
  std::optional<int64_t> size;
  std::string symbolName; // Symbol name for SymbolRefAttr references
};

struct HardwareInfo {
  llvm::SmallVector<SpatialDimInfo> spatialDimInfoVec;
  bool hasBidirInterconnect = false;
  int64_t l1Size = 0;

  bool skipPermutation() const {
    return spatialDimInfoVec.size() == 2 && hasBidirInterconnect;
  }
};

typedef llvm::SmallVector<llvm::SmallVector<unsigned>> DimBuckets;

/**
 * \brief Collect hardware info from a DF module.
 */
mlir::LogicalResult GetHardwareInfoForExploration(mlir::ModuleOp dfModule,
                                                  HardwareInfo &hardwareInfo);

/**
 * \brief Enumerate all unique mappings and emit one function clone per mapping.
 */
mlir::OwningOpRef<mlir::ModuleOp>
EnumerateSpatialMappings(mlir::ModuleOp affineModule,
                         const HardwareInfo &hardwareInfo);

/**
 * \brief Enumerate mappings from Triton-shared grid dims to hardware spatial
 * dims.
 */
mlir::OwningOpRef<mlir::ModuleOp>
enumerateTritonSharedSpatialMappings(mlir::ModuleOp module,
                                     const HardwareInfo &hardwareInfo,
                                     unsigned numGridDims = 3);

} // namespace loom
