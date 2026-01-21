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

struct MatUnitInfo {
  llvm::SmallVector<int64_t, 3> shape; // [BM, BN, BK] alignment granularity
  std::string name;
};

struct HardwareInfo {
  llvm::SmallVector<SpatialDimInfo> spatialDimInfoVec;
  bool hasBidirInterconnect = false;
  int64_t l1Size = 0;
  llvm::SmallVector<MatUnitInfo> matUnits; // Matrix unit info for alignment
  int64_t matUnitCount = 0; // Number of mat_units per core for pipeline

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
} // namespace loom
