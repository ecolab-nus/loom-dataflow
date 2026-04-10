#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "LoomDialect.h.inc"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace loom;

namespace {

/// Checks if an op is effectively "between" two other ops in control flow.
/// This is a conservative check for ops within the same block.
bool isInterveningUsage(Operation *start, Operation *end, Value memref) {
  if (start->getBlock() != end->getBlock())
    return false;

  // Simple check: for each user of the memref, check its position.
  for (Operation *user : memref.getUsers()) {
    if (user == start || user == end)
      continue;

    // Allowed users that are typically after the sequence or semantically safe.
    if (isa<loom::SemaphoreGiveOp>(user))
      continue;
    if (auto copyOp = dyn_cast<loom::CopyOp>(user)) {
      if (copyOp.getSource() == memref)
        continue;
    }

    // Common pattern in this project: if in the same block, we check order.
    if (user->getBlock() == start->getBlock()) {
      if (start->isBeforeInBlock(user) && user->isBeforeInBlock(end)) {
        return true;
      }
    }
  }
  return false;
}

template <typename LinalgMatmulOp, typename LoomMatmulOp>
void processLinalgMatmul(LinalgMatmulOp matmulOp,
                         SmallPtrSetImpl<Operation *> &fillsToErase) {
  if (matmulOp.getNumDpsInits() != 1)
    return;

  Value outs = matmulOp.getDpsInits()[0];

  // Find a preceding linalg.fill(0) for this memref.
  linalg::FillOp fillOp = nullptr;
  for (Operation *user : outs.getUsers()) {
    auto candidate = dyn_cast<linalg::FillOp>(user);
    if (!candidate || candidate.getOutputs().size() != 1 ||
        candidate.getOutputs()[0] != outs)
      continue;

    Value fillVal = candidate.getInputs()[0];
    if (matchPattern(fillVal, m_AnyZeroFloat()) ||
        matchPattern(fillVal, m_Zero())) {
      // Check if this fill immediately precedes this matmul without intervening
      // usage.
      if (!isInterveningUsage(candidate, matmulOp, outs)) {
        fillOp = candidate;
        break;
      }
    }
  }

  if (!fillOp)
    return;

  // Pattern matched: linalg.fill(0, outs) exists and matmul uses outs.
  OpBuilder builder(matmulOp);
  LoomMatmulOp::create(builder, matmulOp.getLoc(), matmulOp.getInputs()[0],
                       matmulOp.getInputs()[1], outs);

  matmulOp.erase();
  fillsToErase.insert(fillOp);
}

struct FuseZeroFillMatmulPass
    : public PassWrapper<FuseZeroFillMatmulPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseZeroFillMatmulPass)

  StringRef getArgument() const override { return "tt-fuse-zero-fill-matmul"; }

  StringRef getDescription() const override {
    return "Fuse linalg.fill(0) and linalg.matmul/batch_matmul into "
           "loom.matmul/batch_matmul if no intervening usage exists.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      SmallPtrSet<Operation *, 4> fillsToErase;

      // 1. Process regular matmuls
      SmallVector<linalg::MatmulOp> matmuls;
      funcOp.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
      for (auto op : matmuls)
        processLinalgMatmul<linalg::MatmulOp, loom::MatmulOp>(op, fillsToErase);

      // 2. Process batch matmuls
      SmallVector<linalg::BatchMatmulOp> batchMatmuls;
      funcOp.walk(
          [&](linalg::BatchMatmulOp op) { batchMatmuls.push_back(op); });
      for (auto op : batchMatmuls)
        processLinalgMatmul<linalg::BatchMatmulOp, loom::BatchMatmulOp>(
            op, fillsToErase);

      for (auto *op : fillsToErase) {
        if (op->use_empty())
          op->erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createFuseZeroFillMatmulPass() {
  return std::make_unique<FuseZeroFillMatmulPass>();
}
