#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
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

static Value traceToSemaphore(Value value) {
  if (!value)
    return nullptr;

  llvm::SmallPtrSet<Value, 32> visited;
  llvm::SmallVector<Value, 32> worklist = {value};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    if (auto semTake = current.getDefiningOp<loom::SemaphoreTakeOp>())
      return semTake.getResult();

    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      Operation *parent = blockArg.getOwner()->getParentOp();

      if (auto scfFor = dyn_cast<scf::ForOp>(parent)) {
        unsigned argNum = blockArg.getArgNumber();
        if (argNum > 0 && argNum - 1 < scfFor.getInits().size())
          worklist.push_back(scfFor.getInits()[argNum - 1]);
      } else if (auto affFor = dyn_cast<affine::AffineForOp>(parent)) {
        unsigned bodyArgs = affFor.getBody()->getNumArguments();
        unsigned iterCount = affFor.getNumIterOperands();
        unsigned firstIterArg = bodyArgs - iterCount;
        unsigned argNum = blockArg.getArgNumber();
        if (argNum >= firstIterArg) {
          unsigned iterIdx = argNum - firstIterArg;
          if (iterIdx < affFor.getInits().size())
            worklist.push_back(affFor.getInits()[iterIdx]);
        }
      } else if (auto affPar = dyn_cast<affine::AffineParallelOp>(parent)) {
        unsigned argNum = blockArg.getArgNumber();
        if (argNum < affPar.getInits().size())
          worklist.push_back(affPar.getInits()[argNum]);
      }
      continue;
    }

    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;

    if (auto init = dyn_cast<loom::InitTensorOp>(defOp)) {
      worklist.push_back(init.getBuffer());
      continue;
    }
    if (auto copyToTensor = dyn_cast<loom::CopyToTensorOp>(defOp)) {
      worklist.push_back(copyToTensor.getBuffer());
      continue;
    }
    if (auto toTensor = dyn_cast<loom::BufferizeToTensorOp>(defOp)) {
      worklist.push_back(toTensor.getSource());
      continue;
    }
    if (auto toMemref = dyn_cast<loom::BufferizeToMemrefOp>(defOp)) {
      worklist.push_back(toMemref.getSource());
      continue;
    }
    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(defOp)) {
      worklist.push_back(toTensor.getBuffer());
      continue;
    }
    if (auto toBuffer = dyn_cast<bufferization::ToBufferOp>(defOp)) {
      worklist.push_back(toBuffer.getTensor());
      continue;
    }
    if (auto mcast = dyn_cast<memref::CastOp>(defOp)) {
      worklist.push_back(mcast.getSource());
      continue;
    }
    if (auto tcast = dyn_cast<tensor::CastOp>(defOp)) {
      worklist.push_back(tcast.getSource());
      continue;
    }

    if (auto dps = dyn_cast<DestinationStyleOpInterface>(defOp)) {
      if (auto res = dyn_cast<OpResult>(current)) {
        unsigned resIdx = res.getResultNumber();
        ValueRange inits = dps.getDpsInits();
        if (resIdx < inits.size()) {
          worklist.push_back(inits[resIdx]);
          continue;
        }
      }
    }

    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }

  return nullptr;
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
    DominanceInfo dominance(module);

    SmallVector<WriteOutChain, 16> chains;
    bool hadError = false;

    module.walk([&](Operation *op) {
      if (auto gatherOp = dyn_cast<loom::GatherOp>(op)) {
        if (auto chain = buildGatherChain(gatherOp))
          chains.push_back(*chain);
        if (auto chain = buildGatherFirstConsumerChain(gatherOp))
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

      bool rewiredAnyUse = false;
      Value targetTensor = chain.targetTensor;
      targetTensor.replaceUsesWithIf(
          sync->getResult(0), [&](OpOperand &use) {
            Operation *owner = use.getOwner();
            if (owner == sync.getOperation())
              return false;
            if (dominance.dominates(sync.getOperation(), owner)) {
              rewiredAnyUse = true;
              return true;
            }
            return false;
          });
      if (!rewiredAnyUse) {
        chain.first()->emitError(
            "failed to rewire any dominated use of sync target tensor");
        signalPassFailure();
        return;
      }

      Value originalSemaphore = traceToSemaphore(chain.targetTensor);
      if (!originalSemaphore) {
        chain.first()->emitError(
            "failed to trace write-out chain target to source semaphore");
        signalPassFailure();
        return;
      }

      loom::SemaphoreGiveOp originalGive;
      for (Operation *user : originalSemaphore.getUsers()) {
        if (auto semGive = dyn_cast<loom::SemaphoreGiveOp>(user)) {
          originalGive = semGive;
          break;
        }
      }
      if (!originalGive) {
        chain.first()->emitError(
            "failed to find semaphore_give for write-out chain source semaphore");
        signalPassFailure();
        return;
      }

      OpBuilder giveBuilder(originalGive);
      loom::SemaphoreGiveOp::create(giveBuilder, originalGive.getLoc(),
                                    semTake.getResult());
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createWriteOutSyncInsertionPass() {
  return std::make_unique<WriteOutSyncInsertionPass>();
}
