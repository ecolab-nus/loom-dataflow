#include "hardware_info.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "ADLDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"

using namespace mlir;

namespace loom {

LogicalResult GetHardwareInfoForExploration(mlir::ModuleOp hwModule,
                                            HardwareInfo &hardwareInfo) {
  // Find all adl.arch.scale ops in the module. Assert exactly one exists.
  llvm::SmallVector<adl::ArchScaleOp> scaleOps;
  hwModule.walk([&](adl::ArchScaleOp op) { scaleOps.push_back(op); });

  if (scaleOps.size() != 1) {
    hwModule.emitError()
        << "expected exactly one adl.arch.scale op, found " << scaleOps.size();
    return failure();
  }

  adl::ArchScaleOp scaleOp = scaleOps[0];

  // Extract spatial dimensions from the arch.scale operands.
  for (Value dimValue : scaleOp.getSpatialDims()) {
    auto sdOp = dimValue.getDefiningOp<adl::SpatialDimOp>();
    if (!sdOp) {
      scaleOp.emitError()
          << "spatial dimension operand is not defined by adl.spatial_dim";
      return failure();
    }

    SpatialDimInfo info;
    info.name = sdOp.getSymName().str();
    info.symbolName = sdOp.getSymName().str();
    uint64_t sz = sdOp.getSize();
    if (sz > 0)
      info.size = static_cast<int64_t>(sz);
    else
      info.size = std::nullopt;
    hardwareInfo.spatialDimInfoVec.push_back(std::move(info));
  }

  return success();
}

} // namespace loom
