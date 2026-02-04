#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Accumulation Operator Support (Extensibility Interface)
//===----------------------------------------------------------------------===//

/// Returns true if the given operation is a supported accumulation operator.
/// TODO: Extend to support arith.addi, arith.maximumf, arith.mulf, etc.
static bool isSupportedAccumulationOp(Operation *op) {
  return isa<arith::AddFOp>(op);
}

/// Returns true if the given value is the neutral element for the accumulation
/// operator type.
/// TODO: Extend to support neutral elements for other operators:
///   - arith.addi: 0
///   - arith.maximumf: -inf
///   - arith.mulf: 1
static bool isNeutralElement(Value fillValue, Operation *accumulationOp) {
  if (isa<arith::AddFOp>(accumulationOp)) {
    if (auto constOp = fillValue.getDefiningOp<arith::ConstantFloatOp>()) {
      return constOp.value().isZero();
    }
    if (auto constOp = fillValue.getDefiningOp<arith::ConstantOp>()) {
      if (auto floatAttr = dyn_cast_or_null<FloatAttr>(constOp.getValue())) {
        return floatAttr.getValue().isZero();
      }
    }
  }
  // TODO: Handle other accumulation operators here
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

/// Check if producer and consumer have semantically aligned accumulation.
static bool isSemanticAligned(linalg::LinalgOp producer,
                              linalg::GenericOp consumer) {
  Operation *consumerAccOp = getAccumulationOp(consumer);
  Operation *producerRedOp = getReductionOp(producer);

  if (!consumerAccOp || !producerRedOp)
    return false;

  // Both must be the same type of accumulation operator
  return consumerAccOp->getName() == producerRedOp->getName();
}

/// Get the actual accumulator (state variable) from consumer inputs,
/// excluding the producer's result.
static Value getActualAccumulator(linalg::GenericOp consumer,
                                  Value producerResult) {
  for (Value input : consumer.getInputs()) {
    if (input != producerResult)
      return input;
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

    // Step 2: Consumer must have exactly 2 inputs and 1 output (binary op)
    if (consumer.getNumDpsInputs() != 2 || consumer.getNumDpsInits() != 1)
      return failure();

    // Step 3: One input must come from a LinalgOp producer
    linalg::LinalgOp producer = nullptr;
    Value producerResult;
    for (Value input : consumer.getInputs()) {
      if (auto defOp = input.getDefiningOp<linalg::LinalgOp>()) {
        producer = defOp;
        producerResult = input;
        break;
      }
    }
    if (!producer)
      return failure();

    // Step 4: Producer must have single use (the consumer)
    // We only specialize if the result is only used by this accumulation
    if (!producerResult.hasOneUse())
      return failure();

    // Step 5: Producer's outs must come from linalg.fill
    if (producer.getDpsInits().size() != 1)
      return failure();

    Value producerOuts = producer.getDpsInitOperand(0)->get();
    auto fillOp = producerOuts.getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return failure();

    // Step 6: Semantic alignment check
    if (!isSemanticAligned(producer, consumer))
      return failure();

    // Step 7: Neutral element verification
    Operation *consumerAccOp = getAccumulationOp(consumer);
    Value fillValue = fillOp.getInputs()[0];
    if (!isNeutralElement(fillValue, consumerAccOp))
      return failure();

    // Step 8: Get the actual accumulator (typically iter_args)
    Value actualAccumulator = getActualAccumulator(consumer, producerResult);
    if (!actualAccumulator)
      return failure();

    // Step 9: Transformation - redirect producer's outs to actual accumulator
    rewriter.modifyOpInPlace(producer, [&]() {
      producer.getDpsInitOperand(0)->set(actualAccumulator);
    });

    // Step 10: Replace all uses of consumer result with producer result
    rewriter.replaceOp(consumer, producer->getResults());

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
