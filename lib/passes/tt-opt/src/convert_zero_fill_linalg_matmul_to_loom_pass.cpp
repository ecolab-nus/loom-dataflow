#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "LoomDialect.h.inc"
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

bool hasInterveningUse(Operation *fillOp, Operation *targetOp, Value buffer) {
  if (fillOp->getBlock() != targetOp->getBlock())
    return true;
  if (!fillOp->isBeforeInBlock(targetOp))
    return true;

  for (Operation *user : buffer.getUsers()) {
    if (user == fillOp || user == targetOp)
      continue;

    if (user->getBlock() == fillOp->getBlock() &&
        fillOp->isBeforeInBlock(user) && user->isBeforeInBlock(targetOp))
      return true;
  }

  return false;
}

linalg::FillOp findSameBlockZeroFill(Operation *targetOp, Value init) {
  if (auto definingFill = init.getDefiningOp<linalg::FillOp>()) {
    if (definingFill.getOutputs().size() != 1)
      return nullptr;
    if (!isZeroConstant(definingFill.getInputs()[0]))
      return nullptr;
    if (hasInterveningUse(definingFill, targetOp, init))
      return nullptr;
    return definingFill;
  }

  for (Operation *user : init.getUsers()) {
    auto fillOp = dyn_cast<linalg::FillOp>(user);
    if (!fillOp || fillOp.getOutputs().size() != 1 ||
        fillOp.getOutputs()[0] != init)
      continue;

    if (!isZeroConstant(fillOp.getInputs()[0]))
      continue;

    if (!hasInterveningUse(fillOp, targetOp, init))
      return fillOp;
  }

  return nullptr;
}

bool areMemRefs(Value lhs, Value rhs, Value out) {
  return isa<MemRefType>(lhs.getType()) && isa<MemRefType>(rhs.getType()) &&
         isa<MemRefType>(out.getType());
}

Value getFillInitForOutput(linalg::FillOp fillOp, Value targetInit) {
  if (fillOp.getNumResults() > 0 && fillOp.getResult(0) == targetInit)
    return fillOp.getOutputs()[0];
  return targetInit;
}

template <typename LinalgMatmulOp, typename LoomMatmulOp>
bool convertMatmul(LinalgMatmulOp matmulOp, RewriterBase &rewriter) {
  if (matmulOp->getNumResults() != 0 || matmulOp.getNumDpsInputs() != 2 ||
      matmulOp.getNumDpsInits() != 1)
    return false;

  Value lhs = matmulOp.getDpsInputs()[0];
  Value rhs = matmulOp.getDpsInputs()[1];
  Value targetInit = matmulOp.getDpsInits()[0];
  linalg::FillOp fillOp = findSameBlockZeroFill(matmulOp, targetInit);
  if (!fillOp)
    return false;

  Value output = getFillInitForOutput(fillOp, targetInit);
  if (!areMemRefs(lhs, rhs, output))
    return false;

  rewriter.setInsertionPoint(matmulOp);
  rewriter.create<LoomMatmulOp>(matmulOp.getLoc(), lhs, rhs, output);
  rewriter.eraseOp(matmulOp);
  return true;
}

struct ConvertZeroFillLinalgMatmulToLoomPass
    : public PassWrapper<ConvertZeroFillLinalgMatmulToLoomPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertZeroFillLinalgMatmulToLoomPass)

  StringRef getArgument() const override {
    return "tt-convert-zero-fill-linalg-matmul-to-loom";
  }

  StringRef getDescription() const override {
    return "Convert zero-initialized linalg.matmul and linalg.batch_matmul "
           "ops to Loom matmul ops.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(module.getContext());

    SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
        candidates.push_back(op);
    });

    for (Operation *op : candidates) {
      if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
        (void)convertMatmul<linalg::MatmulOp, loom::MatmulOp>(matmulOp,
                                                              rewriter);
        continue;
      }

      if (auto batchMatmulOp = dyn_cast<linalg::BatchMatmulOp>(op)) {
        (void)convertMatmul<linalg::BatchMatmulOp, loom::BatchMatmulOp>(
            batchMatmulOp, rewriter);
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createConvertZeroFillLinalgMatmulToLoomPass() {
  return std::make_unique<ConvertZeroFillLinalgMatmulToLoomPass>();
}
