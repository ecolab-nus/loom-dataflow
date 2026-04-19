#include "Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include "ssa_utils.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

class GatherSyncInsertionPass
    : public PassWrapper<GatherSyncInsertionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GatherSyncInsertionPass)

  StringRef getArgument() const override { return "loom-gather-sync-insertion"; }

  StringRef getDescription() const override {
    return "Insert loom.sync and semaphore lifecycle around loom.gather ins";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<loom::GatherOp, 8> gathers;
    module.walk([&](loom::GatherOp gatherOp) {
      if (isa<RankedTensorType>(gatherOp.getIns().getType()))
        gathers.push_back(gatherOp);
    });

    for (loom::GatherOp gatherOp : gathers) {
      Value oldIns = gatherOp.getIns();
      auto insTensorType = dyn_cast<RankedTensorType>(oldIns.getType());
      if (!insTensorType) {
        gatherOp.emitError("expected tensor-mode loom.gather");
        signalPassFailure();
        return;
      }

      Value alloc = loom::utils::traceToRootAlloc(oldIns);
      if (!alloc) {
        gatherOp.emitError("failed to trace gather ins to root loom.alloc");
        signalPassFailure();
        return;
      }

      auto allocOp = alloc.getDefiningOp<loom::AllocOp>();
      if (!allocOp) {
        gatherOp.emitError("traceToRootAlloc result is not from loom.alloc");
        signalPassFailure();
        return;
      }

      auto allocType = dyn_cast<MemRefType>(alloc.getType());
      if (!allocType) {
        gatherOp.emitError("root alloc result is not a memref");
        signalPassFailure();
        return;
      }

      if (allocType.getRank() != insTensorType.getRank()) {
        gatherOp.emitError("alloc rank does not match gather ins rank");
        signalPassFailure();
        return;
      }

      SmallVector<int64_t, 4> staticSizes(insTensorType.getShape().begin(),
                                          insTensorType.getShape().end());

      SmallVector<Value, 4> dynamicSizes;
      auto allocDyn = allocOp.getSizes();
      unsigned allocDynIdx = 0;
      for (int64_t dim : allocType.getShape()) {
        if (dim == ShapedType::kDynamic) {
          if (allocDynIdx >= allocDyn.size()) {
            gatherOp.emitError("alloc dynamic sizes are inconsistent");
            signalPassFailure();
            return;
          }
          dynamicSizes.push_back(allocDyn[allocDynIdx++]);
        }
      }

      unsigned expectedDyn = llvm::count_if(
          staticSizes, [](int64_t d) { return d == ShapedType::kDynamic; });
      if (dynamicSizes.size() < expectedDyn) {
        gatherOp.emitError("ins tensor expects more dynamic sizes than alloc provides");
        signalPassFailure();
        return;
      }
      if (dynamicSizes.size() > expectedDyn)
        dynamicSizes.resize(expectedDyn);

      OpBuilder builder(gatherOp);
      auto semTake = loom::SemaphoreTakeOp::create(builder, gatherOp.getLoc(),
                                                   allocType, alloc);
      auto initTensor = loom::InitTensorOp::create(
          builder, gatherOp.getLoc(), insTensorType, semTake.getResult(),
          dynamicSizes, builder.getDenseI64ArrayAttr(staticSizes));
      auto sync = loom::SyncOp::create(builder, gatherOp.getLoc(),
                                       TypeRange{insTensorType}, oldIns,
                                       initTensor.getResult());

      gatherOp.getInsMutable().assign(sync->getResult(0));

      OpBuilder afterGatherBuilder(gatherOp->getContext());
      afterGatherBuilder.setInsertionPointAfter(gatherOp);
      loom::SemaphoreGiveOp::create(afterGatherBuilder, gatherOp.getLoc(),
                                    semTake.getResult());
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createGatherSyncInsertionPass() {
  return std::make_unique<GatherSyncInsertionPass>();
}
