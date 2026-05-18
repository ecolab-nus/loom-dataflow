// Guarded elementwise fusion pass.
//
// Mirrors mlir::createLinalgElementwiseOpFusionPass but blocks fusion whenever
// the producer OR consumer generic has a leading (outermost) reduction iterator,
// or when it would fuse a single binary elementwise generic into another single
// binary elementwise generic. Such fused forms are not processable or profitable
// on the target hardware.

#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// Returns true if 'op' is a linalg.generic whose outermost iterator is a
// reduction — i.e. iterator_types[0] == "reduction".
static bool hasLeadingReduction(Operation *op) {
  auto genericOp = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!genericOp)
    return false;
  auto iters = genericOp.getIteratorTypesArray();
  return !iters.empty() && iters.front() == utils::IteratorType::reduction;
}

static bool isScalarType(Type type) { return type.isIntOrIndexOrFloat(); }

static bool isBinaryScalarOp(Operation *op) {
  if (!op || op->getNumOperands() != 2 || op->getNumResults() != 1)
    return false;

  if (!llvm::all_of(op->getOperandTypes(), isScalarType))
    return false;
  if (!isScalarType(op->getResult(0).getType()))
    return false;

  return OpTrait::hasElementwiseMappableTraits(op);
}

static Operation *getYieldedBinaryScalarOp(linalg::GenericOp genericOp) {
  if (genericOp.getRegion().empty())
    return nullptr;

  auto yieldOp =
      dyn_cast<linalg::YieldOp>(genericOp.getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return nullptr;

  Operation *yieldedOp = yieldOp.getOperand(0).getDefiningOp();
  return isBinaryScalarOp(yieldedOp) ? yieldedOp : nullptr;
}

static Operation *getFirstBinaryScalarOp(linalg::GenericOp genericOp) {
  if (genericOp.getRegion().empty())
    return nullptr;

  Block &body = genericOp.getRegion().front();
  if (body.empty() || isa<linalg::YieldOp>(body.front()))
    return nullptr;

  Operation *firstOp = &body.front();
  return isBinaryScalarOp(firstOp) ? firstOp : nullptr;
}

static bool isOperandFromGenericInput(Operation *payloadOp, unsigned operandIdx,
                                      linalg::GenericOp consumer,
                                      Value &inputTensor) {
  if (!payloadOp || operandIdx >= payloadOp->getNumOperands())
    return false;

  auto blockArg = dyn_cast<BlockArgument>(payloadOp->getOperand(operandIdx));
  if (!blockArg || blockArg.getOwner() != consumer.getBody())
    return false;

  unsigned argNumber = blockArg.getArgNumber();
  if (argNumber >= consumer.getNumDpsInputs())
    return false;

  inputTensor = consumer.getInputs()[argNumber];
  return true;
}

static bool wouldCreateBinaryScalarChain(linalg::GenericOp consumer) {
  if (!consumer.isAllParallelLoops())
    return false;

  Operation *firstBinaryOp = getFirstBinaryScalarOp(consumer);
  if (!firstBinaryOp)
    return false;

  for (unsigned operandIdx = 0; operandIdx < 2; ++operandIdx) {
    Value inputTensor;
    if (!isOperandFromGenericInput(firstBinaryOp, operandIdx, consumer,
                                   inputTensor))
      continue;

    auto producer = inputTensor.getDefiningOp<linalg::GenericOp>();
    if (producer && getYieldedBinaryScalarOp(producer))
      return true;
  }

  return false;
}

struct LinalgGuardedElementwiseOpFusionPass
    : public PassWrapper<LinalgGuardedElementwiseOpFusionPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LinalgGuardedElementwiseOpFusionPass)

  StringRef getArgument() const override {
    return "loom-linalg-guarded-elementwise-fusion";
  }

  StringRef getDescription() const override {
    return "Elementwise op fusion that skips fusions producing a leading-"
           "reduction generic or binary-elementwise chain";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);

    // Block fusion when either the producer or the consumer has a leading
    // reduction iterator, or when the consumer already has a binary scalar
    // payload fed by another binary-yield generic. Fall back to the upstream
    // single-use requirement.
    linalg::ControlFusionFn controlFn = [](OpOperand *fusedOperand) -> bool {
      Operation *producer = fusedOperand->get().getDefiningOp();
      if (!producer || !producer->hasOneUse())
        return false;
      if (hasLeadingReduction(producer))
        return false;
      if (hasLeadingReduction(fusedOperand->getOwner()))
        return false;
      if (auto consumer =
              dyn_cast<linalg::GenericOp>(fusedOperand->getOwner())) {
        if (wouldCreateBinaryScalarChain(consumer))
          return false;
      }
      return true;
    };

    // Mirror the full upstream pattern set so we keep all other beneficial
    // fusions and canonicalizations.
    linalg::populateElementwiseOpsFusionPatterns(patterns, controlFn);
    linalg::populateFoldReshapeOpsByExpansionPatterns(patterns, controlFn);
    tensor::populateBubbleUpExpandShapePatterns(patterns);

    affine::AffineApplyOp::getCanonicalizationPatterns(patterns, context);
    linalg::GenericOp::getCanonicalizationPatterns(patterns, context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
    context->getLoadedDialect<linalg::LinalgDialect>()
        ->getCanonicalizationPatterns(patterns);

    linalg::populateConstantFoldLinalgOperations(patterns, controlFn);

    if (failed(applyPatternsGreedily(
            op, std::move(patterns),
            GreedyRewriteConfig().setUseTopDownTraversal()))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createLinalgGuardedElementwiseOpFusionPass() {
  return std::make_unique<LinalgGuardedElementwiseOpFusionPass>();
}
