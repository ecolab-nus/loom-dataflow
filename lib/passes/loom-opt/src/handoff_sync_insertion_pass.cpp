#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <optional>

#include "utils.h"

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

// TODO: Generalize the write-out-specific chain model as this pass grows to
// cover more execution-core handoff boundaries.

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

static std::optional<WriteOutChain>
buildGatherFirstConsumerChain(loom::GatherOp gatherOp) {
  if (gatherOp->getNumResults() == 0)
    return std::nullopt;
  Value gatherResult = gatherOp->getResult(0);
  if (!isa<RankedTensorType>(gatherResult.getType()))
    return std::nullopt;

  Operation *firstConsumer = nullptr;
  for (Operation *user : gatherResult.getUsers()) {
    if (!firstConsumer) {
      firstConsumer = user;
      continue;
    }
    if (user->getBlock() == firstConsumer->getBlock() &&
        user->isBeforeInBlock(firstConsumer)) {
      firstConsumer = user;
    }
  }
  if (!firstConsumer)
    return std::nullopt;

  WriteOutChain chain;
  chain.ops.push_back(firstConsumer);
  chain.targetTensor = gatherResult;
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

static FailureOr<WriteOutChain> buildWriteBackCopyChain(memref::CopyOp copyOp) {
  auto toBuffer = copyOp.getSource().getDefiningOp<bufferization::ToBufferOp>();
  if (!toBuffer)
    return failure();

  auto tensorType = dyn_cast<RankedTensorType>(toBuffer.getTensor().getType());
  if (!tensorType)
    return failure();

  WriteOutChain chain;
  chain.ops.push_back(toBuffer.getOperation());
  chain.ops.push_back(copyOp.getOperation());
  chain.targetTensor = toBuffer.getTensor();
  return chain;
}

static LogicalResult insertSyncForTensor(Value targetTensor,
                                         Operation *insertionAnchor,
                                         bool insertAfterAnchor,
                                         DominanceInfo &dominance,
                                         StringRef diagnosticName) {
  if (!targetTensor)
    return success();

  auto tensorType = dyn_cast<RankedTensorType>(targetTensor.getType());
  if (!tensorType)
    return insertionAnchor->emitError()
           << diagnosticName << " target must be ranked tensor";

  OpBuilder builder(insertionAnchor);
  if (insertAfterAnchor)
    builder.setInsertionPointAfter(insertionAnchor);
  else
    builder.setInsertionPoint(insertionAnchor);

  SmallVector<loom::utils::SymbolicDim, 4> tracedShape =
      loom::utils::traceShape(targetTensor);

  SmallVector<Value, 4> dynamicSizes;
  for (auto [idx, dim] : llvm::enumerate(tensorType.getShape())) {
    if (dim != ShapedType::kDynamic)
      continue;

    Value dynSizeVal = nullptr;
    if (idx < tracedShape.size()) {
      if (auto tracedVal = llvm::dyn_cast<Value>(tracedShape[idx])) {
        Operation *def = tracedVal.getDefiningOp();
        if (!def || dominance.dominates(def, insertionAnchor))
          dynSizeVal = tracedVal;
      } else if (auto tracedAttr = llvm::dyn_cast<Attribute>(tracedShape[idx])) {
        if (auto intAttr = dyn_cast<IntegerAttr>(tracedAttr)) {
          if (intAttr.getInt() >= 0) {
            dynSizeVal = arith::ConstantIndexOp::create(
                builder, insertionAnchor->getLoc(), intAttr.getInt());
          }
        }
      }
    }

    if (!dynSizeVal) {
      insertionAnchor->emitError()
          << diagnosticName << " could not resolve dynamic dimension " << idx
          << " for sync init tensor without falling back to tensor.dim";
      assert(false && "unresolved loom.sync init dynamic dimension");
      llvm::report_fatal_error(
          "unresolved loom.sync init dynamic dimension");
    }
    dynamicSizes.push_back(dynSizeVal);
  }

  auto empty = tensor::EmptyOp::create(builder, insertionAnchor->getLoc(),
                                       tensorType.getShape(),
                                       tensorType.getElementType(),
                                       dynamicSizes);
  auto sync = loom::SyncOp::create(builder, insertionAnchor->getLoc(),
                                   TypeRange{tensorType}, targetTensor,
                                   empty.getResult());

  bool rewiredAnyUse = false;
  targetTensor.replaceUsesWithIf(sync->getResult(0), [&](OpOperand &use) {
    Operation *owner = use.getOwner();
    if (owner == sync.getOperation())
      return false;
    if (dominance.dominates(sync.getOperation(), owner)) {
      rewiredAnyUse = true;
      return true;
    }
    return false;
  });
  (void)rewiredAnyUse;
  return success();
}

static LogicalResult handleToTensor(bufferization::ToTensorOp toTensorOp,
                                    DominanceInfo &dominance) {
  if (!isa<RankedTensorType>(toTensorOp.getResult().getType()))
    return success();

  return insertSyncForTensor(toTensorOp.getResult(), toTensorOp.getOperation(),
                             /*insertAfterAnchor=*/true, dominance,
                             "bufferization.to_tensor handoff");
}

class HandoffSyncInsertionPass
    : public PassWrapper<HandoffSyncInsertionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HandoffSyncInsertionPass)

  StringRef getArgument() const override {
    return "loom-handoff-sync-insertion";
  }

  StringRef getDescription() const override {
    return "Insert tensor.empty + loom.sync around execution-core handoffs";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    DominanceInfo dominance(module);

    SmallVector<WriteOutChain, 16> chains;
    SmallVector<bufferization::ToTensorOp, 16> toTensorOps;
    bool hadError = false;

    module.walk([&](Operation *op) {
      if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(op)) {
        toTensorOps.push_back(toTensorOp);
        return;
      }
      if (auto gatherOp = dyn_cast<loom::GatherOp>(op)) {
        if (auto chain = buildGatherChain(gatherOp))
          chains.push_back(*chain);
        if (auto chain = buildGatherFirstConsumerChain(gatherOp))
          chains.push_back(*chain);
        return;
      }
      if (auto memCopyOp = dyn_cast<memref::CopyOp>(op)) {
        auto chainOr = buildWriteBackCopyChain(memCopyOp);
        if (succeeded(chainOr))
          chains.push_back(*chainOr);
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
      if (failed(insertSyncForTensor(chain.targetTensor, chain.first(),
                                     /*insertAfterAnchor=*/false, dominance,
                                     "write-out chain"))) {
        signalPassFailure();
        return;
      }
    }

    for (bufferization::ToTensorOp toTensorOp : toTensorOps) {
      if (failed(handleToTensor(toTensorOp, dominance))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createHandoffSyncInsertionPass() {
  return std::make_unique<HandoffSyncInsertionPass>();
}
