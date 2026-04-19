#include "Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

#include "ssa_utils.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

struct WriteOutChain {
  SmallVector<Operation *, 2> ops;
  Value targetTensor;

  Operation *first() const { return ops.front(); }
  Operation *last() const { return ops.back(); }
};

static bool isL1ToDramCopy(loom::CopyOp copyOp) {
  auto src = copyOp.getSrcMemSpaceAttr();
  auto dst = copyOp.getDstMemSpaceAttr();
  if (!src || !dst)
    return false;

  return src.getLeafReference() == "mem_L1" &&
         dst.getLeafReference() == "mem_DRAM";
}

static std::optional<WriteOutChain> buildGatherChain(loom::GatherOp gatherOp) {
  if (!isa<RankedTensorType>(gatherOp.getIns().getType()))
    return std::nullopt;

  WriteOutChain chain;
  chain.ops.push_back(gatherOp.getOperation());
  chain.targetTensor = gatherOp.getIns();
  return chain;
}

static FailureOr<WriteOutChain> buildL1ToDramCopyChain(loom::CopyOp copyOp) {
  if (!isL1ToDramCopy(copyOp))
    return failure();

  auto b2m = copyOp.getSource().getDefiningOp<loom::BufferizeToMemrefOp>();
  if (!b2m) {
    copyOp.emitError("expected source of L1->DRAM loom.copy to be "
                     "loom.bufferize_to_memref");
    return failure();
  }

  auto tensorType = dyn_cast<RankedTensorType>(b2m.getSource().getType());
  if (!tensorType) {
    b2m.emitError("expected loom.bufferize_to_memref source to be ranked tensor");
    return failure();
  }

  WriteOutChain chain;
  chain.ops.push_back(b2m.getOperation());
  chain.ops.push_back(copyOp.getOperation());
  chain.targetTensor = b2m.getSource();
  return chain;
}

class WriteOutSyncInsertionPass
    : public PassWrapper<WriteOutSyncInsertionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WriteOutSyncInsertionPass)

  StringRef getArgument() const override {
    return "loom-writeout-sync-insertion";
  }

  StringRef getDescription() const override {
    return "Insert loom.sync and semaphore lifecycle around write-out chains";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<WriteOutChain, 16> chains;
    bool hadError = false;

    module.walk([&](Operation *op) {
      if (auto gatherOp = dyn_cast<loom::GatherOp>(op)) {
        if (auto chain = buildGatherChain(gatherOp))
          chains.push_back(*chain);
        return;
      }
      if (auto copyOp = dyn_cast<loom::CopyOp>(op)) {
        auto chainOr = buildL1ToDramCopyChain(copyOp);
        if (succeeded(chainOr)) {
          chains.push_back(*chainOr);
          return;
        }
        if (isL1ToDramCopy(copyOp))
          hadError = true;
      }
    });
    if (hadError) {
      signalPassFailure();
      return;
    }

    for (const WriteOutChain &chain : chains) {
      if (!chain.targetTensor)
        continue;

      auto tensorType = dyn_cast<RankedTensorType>(chain.targetTensor.getType());
      if (!tensorType) {
        chain.first()->emitError("write-out chain target must be ranked tensor");
        signalPassFailure();
        return;
      }

      Value alloc = loom::utils::traceToRootAlloc(chain.targetTensor);
      if (!alloc) {
        chain.first()->emitError(
            "failed to trace write-out chain target to root loom.alloc");
        signalPassFailure();
        return;
      }

      auto allocOp = alloc.getDefiningOp<loom::AllocOp>();
      auto allocType = dyn_cast<MemRefType>(alloc.getType());
      if (!allocOp || !allocType) {
        chain.first()->emitError("traceToRootAlloc must return loom.alloc memref");
        signalPassFailure();
        return;
      }

      if (allocType.getRank() != tensorType.getRank()) {
        chain.first()->emitError(
            "alloc rank does not match write-out chain target tensor rank");
        signalPassFailure();
        return;
      }

      SmallVector<int64_t, 4> staticSizes(tensorType.getShape().begin(),
                                          tensorType.getShape().end());

      SmallVector<Value, 4> dynamicSizes;
      auto allocDyn = allocOp.getSizes();
      unsigned allocDynIdx = 0;
      for (int64_t dim : allocType.getShape()) {
        if (dim == ShapedType::kDynamic) {
          if (allocDynIdx >= allocDyn.size()) {
            chain.first()->emitError("alloc dynamic sizes are inconsistent");
            signalPassFailure();
            return;
          }
          dynamicSizes.push_back(allocDyn[allocDynIdx++]);
        }
      }

      unsigned expectedDyn = llvm::count_if(
          staticSizes, [](int64_t d) { return d == ShapedType::kDynamic; });
      if (dynamicSizes.size() < expectedDyn) {
        chain.first()->emitError(
            "target tensor expects more dynamic sizes than alloc provides");
        signalPassFailure();
        return;
      }
      if (dynamicSizes.size() > expectedDyn)
        dynamicSizes.resize(expectedDyn);

      OpBuilder beforeBuilder(chain.first());
      auto semTake = loom::SemaphoreTakeOp::create(beforeBuilder,
                                                   chain.first()->getLoc(),
                                                   allocType, alloc);
      auto initTensor = loom::InitTensorOp::create(
          beforeBuilder, chain.first()->getLoc(), tensorType,
          semTake.getResult(),
          dynamicSizes, beforeBuilder.getDenseI64ArrayAttr(staticSizes));
      auto sync = loom::SyncOp::create(beforeBuilder, chain.first()->getLoc(),
                                       TypeRange{tensorType}, chain.targetTensor,
                                       initTensor.getResult());

      if (auto gatherOp = dyn_cast<loom::GatherOp>(chain.first())) {
        gatherOp.getInsMutable().assign(sync->getResult(0));
      } else if (auto b2m = dyn_cast<loom::BufferizeToMemrefOp>(chain.first())) {
        b2m.getSourceMutable().assign(sync->getResult(0));
      } else {
        chain.first()->emitError("unsupported chain start op for rewiring");
        signalPassFailure();
        return;
      }

      OpBuilder afterBuilder(chain.last()->getContext());
      afterBuilder.setInsertionPointAfter(chain.last());
      loom::SemaphoreGiveOp::create(afterBuilder, chain.last()->getLoc(),
                                    semTake.getResult());
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createWriteOutSyncInsertionPass() {
  return std::make_unique<WriteOutSyncInsertionPass>();
}
