#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

#include "LoomDialect.h.inc"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace loom;

namespace {

bool isZeroAttribute(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  return false;
}

bool isZeroConstant(Value value) {
  Attribute attr;
  if (matchPattern(value, m_Constant(&attr))) {
    if (isZeroAttribute(attr))
      return true;

    if (auto elementsAttr = dyn_cast<DenseElementsAttr>(attr)) {
      if (!elementsAttr.isSplat())
        return false;
      return isZeroAttribute(elementsAttr.getSplatValue<Attribute>());
    }
  }

  return matchPattern(value, m_AnyZeroFloat()) ||
         matchPattern(value, m_Zero());
}

/// Checks if an op is effectively "between" two other ops in control flow.
/// This is a conservative check for ops within the same block.
bool isInterveningUsage(Operation *start, Operation *end, Value memref) {
  if (start->getBlock() != end->getBlock())
    return true;
  if (!start->isBeforeInBlock(end))
    return true;

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

    if (user->getBlock() == start->getBlock() &&
        start->isBeforeInBlock(user) && user->isBeforeInBlock(end))
      return true;
  }

  return false;
}

linalg::FillOp findZeroFillForOutput(Operation *consumer, Value outs) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(consumer)) {
    if (llvm::is_contained(linalgOp.getDpsInputs(), outs))
      return nullptr;
  }

  for (Operation *user : outs.getUsers()) {
    auto candidate = dyn_cast<linalg::FillOp>(user);
    if (!candidate || candidate.getOutputs().size() != 1 ||
        candidate.getOutputs()[0] != outs)
      continue;

    Value fillVal = candidate.getInputs()[0];
    if (!isZeroConstant(fillVal))
      continue;

    if (!isInterveningUsage(candidate, consumer, outs))
      return candidate;
  }

  return nullptr;
}

template <typename LoomMatmulOp>
void processLoomMatmul(LoomMatmulOp matmulOp,
                       SmallPtrSetImpl<Operation *> &fillsToErase) {
  Value outs = matmulOp.getOuts();

  linalg::FillOp fillOp = findZeroFillForOutput(matmulOp, outs);
  if (!fillOp)
    return;

  fillsToErase.insert(fillOp);
}

struct FoldZeroFillLinalgPass
    : public PassWrapper<FoldZeroFillLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldZeroFillLinalgPass)

  StringRef getArgument() const override { return "tt-fold-zero-fill-linalg"; }

  StringRef getDescription() const override {
    return "Fold same-block linalg.fill(0) ops feeding linalg outs operands.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      SmallPtrSet<Operation *, 4> fillsToErase;

      // Only Loom matmuls created by the preceding conversion have
      // zero-initializing semantics that make the fill redundant.
      SmallVector<loom::MatmulOp> loomMatmuls;
      funcOp.walk([&](loom::MatmulOp op) { loomMatmuls.push_back(op); });
      for (auto op : loomMatmuls)
        processLoomMatmul<loom::MatmulOp>(op, fillsToErase);

      SmallVector<loom::BatchMatmulOp> loomBatchMatmuls;
      funcOp.walk(
          [&](loom::BatchMatmulOp op) { loomBatchMatmuls.push_back(op); });
      for (auto op : loomBatchMatmuls)
        processLoomMatmul<loom::BatchMatmulOp>(op, fillsToErase);

      for (auto *op : fillsToErase) {
        if (op->use_empty())
          op->erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createFoldZeroFillLinalgPass() {
  return std::make_unique<FoldZeroFillLinalgPass>();
}
