/**
 * @file spatial_reuse.cpp
 * @brief Annotate loom.copy area using loom.spatial_mapping reuse analysis.
 */

#include "Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

static bool hasMemSpace(loom::CopyOp copyOp, StringRef src, StringRef dst) {
  SymbolRefAttr srcAttr = copyOp.getSrcMemSpaceAttr();
  SymbolRefAttr dstAttr = copyOp.getDstMemSpaceAttr();
  if (!srcAttr || !dstAttr)
    return false;
  return srcAttr.getLeafReference() == src && dstAttr.getLeafReference() == dst;
}

static loom::SpatialMappingOp findNearestSpatialMapping(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto spatialMapping = dyn_cast<loom::SpatialMappingOp>(parent))
      return spatialMapping;
  }
  return nullptr;
}

static bool dependsOnValue(ArrayRef<Value> roots, Value target) {
  if (!target)
    return false;

  SmallVector<Value, 32> worklist;
  llvm::SmallPtrSet<Value, 32> visited;
  for (Value root : roots) {
    if (root)
      worklist.push_back(root);
  }

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;
    if (current == target)
      return true;

    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;

    for (Value operand : defOp->getOperands()) {
      if (!visited.contains(operand))
        worklist.push_back(operand);
    }
  }

  return false;
}

static bool isSameMixedArea(loom::CopyOp copyOp, ArrayRef<int64_t> staticArea,
                            ArrayRef<Value> dynamicArea) {
  if (!llvm::equal(copyOp.getStaticArea(), staticArea))
    return false;
  return llvm::equal(copyOp.getArea(), dynamicArea);
}

static LogicalResult updateCopyArea(loom::CopyOp copyOp,
                                    loom::SpatialMappingOp spatialMapping) {
  loom::SubviewOp sourceSubview =
      copyOp.getSource().getDefiningOp<loom::SubviewOp>();
  if (!sourceSubview)
    return success();

  SmallVector<Value, 4> offsets(sourceSubview.getOffsets().begin(),
                                sourceSubview.getOffsets().end());
  SmallVector<int64_t, 4> staticArea;
  SmallVector<Value, 4> dynamicArea;
  unsigned reusableAxes = 0;

  auto blockIndices = spatialMapping.getBlockIndices();
  ValueRange lds = spatialMapping.getLds();
  for (auto [blockIdx, ld] : llvm::zip(blockIndices, lds)) {
    if (dependsOnValue(offsets, blockIdx)) {
      staticArea.push_back(1);
      continue;
    }

    staticArea.push_back(ShapedType::kDynamic);
    dynamicArea.push_back(ld);
    ++reusableAxes;
  }

  // Multi-axis spatial reuse is a real pattern, but this pass intentionally
  // rejects it until the compiler has a proper policy for representing and
  // lowering multi-axis multicast.
  if (reusableAxes > 1) {
    copyOp.emitOpError()
        << "has spatial reuse along " << reusableAxes
        << " axes; only one spatial reuse axis is currently supported";
    return failure();
  }

  if (isSameMixedArea(copyOp, staticArea, dynamicArea))
    return success();

  OpBuilder builder(copyOp.getOperation());
  loom::CopyOp::create(builder, copyOp.getLoc(), copyOp.getSource(),
                       copyOp.getDestination(), copyOp.getSrcMemSpaceAttr(),
                       copyOp.getDstMemSpaceAttr(), dynamicArea,
                       builder.getDenseI64ArrayAttr(staticArea),
                       copyOp.getUlX(), copyOp.getUlY(), copyOp.getLrX(),
                       copyOp.getLrY(), copyOp.getReclaimAttr());
  copyOp.erase();
  return success();
}

struct SpatialReusePass
    : public PassWrapper<SpatialReusePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpatialReusePass)

  StringRef getArgument() const override { return "loom-spatial-reuse"; }

  StringRef getDescription() const override {
    return "Annotate loom.copy area from loom.spatial_mapping reuse";
  }

  void runOnOperation() override {
    SmallVector<loom::CopyOp, 16> copyOps;
    getOperation().walk([&](loom::CopyOp copyOp) {
      if (hasMemSpace(copyOp, "mem_DRAM", "mem_L1"))
        copyOps.push_back(copyOp);
    });

    for (loom::CopyOp copyOp : copyOps) {
      loom::SpatialMappingOp spatialMapping =
          findNearestSpatialMapping(copyOp.getOperation());
      if (!spatialMapping)
        continue;
      if (failed(updateCopyArea(copyOp, spatialMapping))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createSpatialReusePass() {
  return std::make_unique<SpatialReusePass>();
}
