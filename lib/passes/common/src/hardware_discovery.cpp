#include "hardware_info.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "DataflowDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.h.inc"
#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

using namespace mlir;

namespace {

static LogicalResult
GetSpatialDimInfo(loom::df::SpatialDimOp sdOp,
                  llvm::SmallVector<loom::SpatialDimInfo> &dimVec) {
  loom::SpatialDimInfo info;
  if (auto nameAttr = sdOp.getSymNameAttr()) {
    info.name = nameAttr.getValue().str();
    info.symbolName = nameAttr.getValue().str();
  } else {
    info.name = "dim";
    info.symbolName = "dim";
  }
  uint64_t sz = sdOp.getSize();
  if (sz > 0)
    info.size = static_cast<int64_t>(sz);
  else
    info.size = std::nullopt;
  dimVec.push_back(std::move(info));
  return success();
}

static std::pair<bool, bool> AnalyzeInterconnectDirection(AffineMap map) {
  if (map.getNumResults() < 2)
    return {false, false};

  bool d0Connected = false;
  bool d1Connected = false;
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    AffineExpr expr = map.getResult(i);
    if (i == 0 && expr != getAffineDimExpr(0, map.getContext()))
      d0Connected = true;
    if (i == 1 && expr != getAffineDimExpr(1, map.getContext()))
      d1Connected = true;
  }
  return {d0Connected, d1Connected};
}

} // namespace

namespace loom {

LogicalResult GetHardwareInfoForExploration(mlir::ModuleOp dfModule,
                                            HardwareInfo &hardwareInfo) {
  bool res = false;
  bool d0Connected = false;
  bool d1Connected = false;
  dfModule.walk([&](Operation *op) {
    if (auto sd = dyn_cast<loom::df::SpatialDimOp>(op)) {
      res =
          res || failed(GetSpatialDimInfo(sd, hardwareInfo.spatialDimInfoVec));
    } else if (auto mem = dyn_cast<loom::df::MemoryOp>(op)) {
      if (mem.getLabel() == "L1") {
        hardwareInfo.l1Size = mem.getSize();
      }
    } else if (auto mat = dyn_cast<loom::df::MatOp>(op)) {
      MatUnitInfo info;
      info.name = mat.getName().str();
      auto shape = mat.getShape();
      for (auto dim : shape)
        info.shape.push_back(dim);
      hardwareInfo.matUnits.push_back(std::move(info));
    } else if (auto core = dyn_cast<loom::df::CoreOp>(op)) {
      if (auto countsAttr = core.getScaleinCountsAttr()) {
        auto counts = countsAttr.asArrayRef();
        if (!counts.empty())
          hardwareInfo.matUnitCount = counts[0];
      }
    } else if (auto ic = dyn_cast<loom::df::InterconnectsOp>(op)) {
      AffineMap map = ic.getMapAttr().getValue();
      auto [x, y] = AnalyzeInterconnectDirection(map);
      if (x && !y) {
        d0Connected = true;
      }
      if (y && !x) {
        d1Connected = true;
      }
      res = res || x || y;
    }
  });
  hardwareInfo.hasBidirInterconnect = d0Connected && d1Connected;
  return success(res);
}

} // namespace loom
