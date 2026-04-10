#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Accumulation Operator Support (Extensibility Interface)
//===----------------------------------------------------------------------===//

static bool isSupportedAccumulationOp(Operation *op) {
  return isa<arith::AddFOp, arith::AddIOp, arith::MaximumFOp, arith::MaxSIOp,
             arith::MinimumFOp, arith::MinSIOp, arith::MulFOp, arith::MulIOp>(
      op);
}

static bool isNeutralElement(Value fillValue, Operation *accumulationOp) {
  Attribute attr;
  if (auto constOp = fillValue.getDefiningOp<arith::ConstantOp>()) {
    attr = constOp.getValue();
  } else if (auto constOp = fillValue.getDefiningOp<arith::ConstantFloatOp>()) {
    attr = constOp.getValueAttr();
  } else if (auto constOp = fillValue.getDefiningOp<arith::ConstantIntOp>()) {
    attr = constOp.getValueAttr();
  }

  if (!attr)
    return false;

  if (isa<arith::AddFOp, arith::AddIOp>(accumulationOp)) {
    if (auto floatAttr = dyn_cast<FloatAttr>(attr))
      return floatAttr.getValue().isZero();
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt() == 0;
  }
  if (isa<arith::MulFOp, arith::MulIOp>(accumulationOp)) {
    if (auto floatAttr = dyn_cast<FloatAttr>(attr))
      return floatAttr.getValue().isExactlyValue(1.0);
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt() == 1;
  }
  if (isa<arith::MaximumFOp, arith::MaxSIOp>(accumulationOp)) {
    if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
      return floatAttr.getValue().isInfinity() &&
             floatAttr.getValue().isNegative();
    }
    // For integer max, we'd need the bitwidth to check for min_int.
    // Skipping for now.
  }
  if (isa<arith::MinimumFOp, arith::MinSIOp>(accumulationOp)) {
    if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
      return floatAttr.getValue().isInfinity() &&
             !floatAttr.getValue().isNegative();
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Semantic Analysis Utilities
//===----------------------------------------------------------------------===//

/// Extract the accumulation operator from a GenericOp consumer.
/// Returns the operation immediately preceding linalg.yield if it's a
/// supported accumulation op, otherwise nullptr.
static Operation *getAccumulationOp(linalg::GenericOp consumer) {
  if (consumer.getRegion().empty())
    return nullptr;
  Block &body = consumer.getRegion().front();
  if (body.empty())
    return nullptr;
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return nullptr;

  Operation *defOp = yieldOp.getOperand(0).getDefiningOp();
  if (defOp && isSupportedAccumulationOp(defOp))
    return defOp;
  return nullptr;
}

/// Extract the reduction operator from a LinalgOp producer.
/// Finds the operation that combines with the outs block argument.
static Operation *getReductionOp(linalg::LinalgOp producer) {
  if (producer->getRegion(0).empty())
    return nullptr;
  Block &body = producer->getRegion(0).front();
  if (body.empty())
    return nullptr;
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return nullptr;

  Value yieldVal = yieldOp.getOperand(0);
  Operation *defOp = yieldVal.getDefiningOp();
  if (!defOp)
    return nullptr;

  // Check which operand references the block argument for outs
  // For LinalgOps, the outs arguments are appended after inputs in block
  // arguments
  unsigned outsIdx = producer.getNumDpsInputs();
  if (body.getNumArguments() <= outsIdx)
    return nullptr;
  BlockArgument outsArg = body.getArgument(outsIdx);

  for (Value operand : defOp->getOperands()) {
    if (operand == outsArg && isSupportedAccumulationOp(defOp)) {
      return defOp;
    }
  }
  return nullptr;
}

/// Returns true if the LinalgOp has at least one reduction iterator.
static bool isReductionProducer(linalg::LinalgOp op) {
  return llvm::any_of(op.getIteratorTypesArray(), [](utils::IteratorType t) {
    return t == utils::IteratorType::reduction;
  });
}

/// Check if producer and consumer have semantically aligned accumulation.
static bool isSemanticAligned(linalg::LinalgOp producer,
                              linalg::GenericOp consumer) {
  if (!isReductionProducer(producer))
    return false;

  Operation *consumerAccOp = getAccumulationOp(consumer);
  if (!consumerAccOp)
    return false;

  Operation *producerRedOp = getReductionOp(producer);
  if (producerRedOp) {
    return consumerAccOp->getName() == producerRedOp->getName();
  }

  return false;
}

/// Get the actual accumulator value from consumer inputs corresponding to the
/// operand of the accumulation op that is NOT the producer result.
static Value getActualAccumulator(linalg::GenericOp consumer,
                                  Value producerResult) {
  Operation *accOp = getAccumulationOp(consumer);
  if (!accOp)
    return Value();

  Block &body = consumer.getRegion().front();
  unsigned producerInputIdx = 0;
  bool found = false;
  for (auto it : llvm::enumerate(consumer.getInputs())) {
    if (it.value() == producerResult) {
      producerInputIdx = it.index();
      found = true;
      break;
    }
  }
  if (!found)
    return Value();
  BlockArgument producerArg = body.getArgument(producerInputIdx);

  // Find the other operand of the accumulation op
  Value otherOperand;
  if (accOp->getOperand(0) == producerArg) {
    otherOperand = accOp->getOperand(1);
  } else if (accOp->getOperand(1) == producerArg) {
    otherOperand = accOp->getOperand(0);
  }

  if (!otherOperand)
    return Value();

  // The other operand must eventually come from a block argument of the generic
  // (if there was fusion, this might be more complex, but we only support
  // cases where it's a direct input for now)
  if (auto blockArg = dyn_cast<BlockArgument>(otherOperand)) {
    if (blockArg.getOwner() == &body &&
        blockArg.getArgNumber() < consumer.getNumDpsInputs()) {
      return consumer.getInputs()[blockArg.getArgNumber()];
    }
  }

  return Value();
}

//===----------------------------------------------------------------------===//
// Main Transformation Pattern
//===----------------------------------------------------------------------===//

struct SpecializeLinalgDestination
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumer,
                                PatternRewriter &rewriter) const override {
    // Step 1: Consumer must be elementwise (all parallel iterators)
    if (!consumer.hasPureTensorSemantics())
      return failure();

    auto iterTypes = consumer.getIteratorTypesArray();
    bool allParallel = llvm::all_of(iterTypes, [](utils::IteratorType t) {
      return t == utils::IteratorType::parallel;
    });
    if (!allParallel)
      return failure();

    // Step 2: Consumer must be a simple accumulation block (acc_op + yield)
    if (consumer.getRegion().front().getOperations().size() != 2)
      return failure();
    if (consumer.getNumDpsInits() != 1)
      return failure();

    // Step 3: Find an alignable LinalgOp producer
    linalg::LinalgOp producer = nullptr;
    Value producerResult;
    for (Value input : consumer.getInputs()) {
      auto defOp = input.getDefiningOp<linalg::LinalgOp>();
      if (!defOp)
        continue;

      // Check if this producer is semantically aligned and its outs is a fill
      if (!isSemanticAligned(defOp, consumer))
        continue;

      if (defOp.getDpsInits().size() != 1)
        continue;
      Value producerOuts = defOp.getDpsInitOperand(0)->get();
      auto fillOp = producerOuts.getDefiningOp<linalg::FillOp>();
      if (!fillOp)
        continue;

      Operation *consumerAccOp = getAccumulationOp(consumer);
      if (!isNeutralElement(fillOp.getInputs()[0], consumerAccOp))
        continue;

      // If we got here, we found our producer
      producer = defOp;
      producerResult = input;
      break;
    }
    if (!producer)
      return failure();

    // Step 4: Get the actual accumulator (typically iter_args)
    Value actualAccumulator = getActualAccumulator(consumer, producerResult);
    if (!actualAccumulator)
      return failure();

    // Step 5: Transformation - clone producer at consumer's location
    rewriter.setInsertionPoint(consumer);

    // Use IRMapping to clone the producer
    // We don't actually need to remap operands during cloning if we update
    // them right after, but it's cleaner to clone the whole thing.
    IRMapping mapper;
    Operation *cloned = rewriter.clone(*producer.getOperation(), mapper);
    auto clonedLinalgOp = cast<linalg::LinalgOp>(cloned);

    // Update the outs operand of the cloned producer to use the actual
    // accumulator
    clonedLinalgOp.getDpsInitOperand(0)->set(actualAccumulator);

    // Step 6: Replace all uses of consumer result with cloned producer result
    rewriter.replaceOp(consumer, cloned->getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct LinalgDestinationSpecializationPass
    : public PassWrapper<LinalgDestinationSpecializationPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LinalgDestinationSpecializationPass)

  StringRef getArgument() const override {
    return "loom-linalg-destination-specialization";
  }

  StringRef getDescription() const override {
    return "Specialize LinalgOp destinations by folding accumulation patterns";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, arith::ArithDialect, func::FuncDialect,
                linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<SpecializeLinalgDestination>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createLinalgDestinationSpecializationPass() {
  return std::make_unique<LinalgDestinationSpecializationPass>();
}
