#include "Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "utils.h"

using namespace mlir;

namespace {

struct FoldRedundantExtractSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {
    // 1. Check all offsets are 0
    for (auto offset : op.getMixedOffsets()) {
      if (auto attr = mlir::dyn_cast<Attribute>(offset)) {
        if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
          if (intAttr.getInt() != 0)
            return failure();
        } else {
          return failure();
        }
      } else {
        // Dynamic offset
        return failure();
      }
    }

    // 2. Check all strides are 1
    for (auto stride : op.getMixedStrides()) {
      if (auto attr = mlir::dyn_cast<Attribute>(stride)) {
        if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
          if (intAttr.getInt() != 1)
            return failure();
        } else {
          return failure();
        }
      } else {
        // Dynamic stride
        return failure();
      }
    }

    // 3a. Check sizes match via SSA value identity (original path)
    Value source = op.getSource();
    auto sourceShape = loom::utils::traceShape(source);
    auto extractSizes = op.getMixedSizes();

    bool ssaMatch = (sourceShape.size() == extractSizes.size());
    if (ssaMatch) {
      for (size_t i = 0; i < sourceShape.size(); ++i) {
        if (sourceShape[i] != extractSizes[i]) {
          ssaMatch = false;
          break;
        }
      }
    }

    if (ssaMatch) {
      rewriter.replaceOp(op, source);
      return success();
    }

    // 3b. Fallback: if source and result types are the same RankedTensorType
    // (identical rank + static dims), this is a full-tensor identity slice.
    // This handles the case where dynamic sizes come from semantically-equal
    // but SSA-distinct Values (e.g. two uses of the same loom.sym).
    auto srcType = mlir::dyn_cast<RankedTensorType>(source.getType());
    auto resType = mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
    if (srcType && resType && srcType == resType) {
      rewriter.replaceOp(op, source);
      return success();
    }

    return failure();
  }
};

struct FoldRedundantExtractSlicePass
    : public PassWrapper<FoldRedundantExtractSlicePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldRedundantExtractSlicePass)

  StringRef getArgument() const override {
    return "loom-fold-redundant-extract-slice";
  }

  StringRef getDescription() const override {
    return "Fold redundant tensor.extract_slice operations where offsets are "
           "0, strides are 1, and sizes match source";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FoldRedundantExtractSlice>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createFoldRedundantExtractSlicePass() {
  return std::make_unique<FoldRedundantExtractSlicePass>();
}
