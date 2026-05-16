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
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
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

// Returns true if 'op' is semantically equivalent to a single binary
// elementwise linalg op. This intentionally delegates the payload and indexing
// map checks to MLIR instead of matching arithmetic op names.
static bool isSingleBinaryElementwiseGeneric(Operation *op) {
  auto genericOp = dyn_cast_or_null<linalg::GenericOp>(op);
  return genericOp && linalg::isaElemwiseSingleBinaryOpInterface(genericOp);
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
    // reduction iterator, or when a single binary elementwise generic feeds
    // another single binary elementwise generic. Fall back to the upstream
    // single-use requirement.
    linalg::ControlFusionFn controlFn = [](OpOperand *fusedOperand) -> bool {
      Operation *producer = fusedOperand->get().getDefiningOp();
      if (!producer || !producer->hasOneUse())
        return false;
      if (hasLeadingReduction(producer))
        return false;
      if (hasLeadingReduction(fusedOperand->getOwner()))
        return false;
      if (isSingleBinaryElementwiseGeneric(producer) &&
          isSingleBinaryElementwiseGeneric(fusedOperand->getOwner()))
        return false;
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
