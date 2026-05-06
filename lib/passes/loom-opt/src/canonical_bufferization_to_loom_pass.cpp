#include "Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

static LogicalResult
deriveTensorSizes(Location loc, Value source, RankedTensorType resultType,
                  PatternRewriter &rewriter, SmallVectorImpl<Value> &dynSizes,
                  SmallVectorImpl<int64_t> &staticSizes) {
  SmallVector<OpFoldResult, 4> mixedSizes;

  if (auto subview = source.getDefiningOp<memref::SubViewOp>()) {
    ArrayRef<int64_t> allStaticSizes = subview.getStaticSizes();
    SmallVector<OpFoldResult, 4> allMixedSizes = subview.getMixedSizes();
    if (allStaticSizes.size() != allMixedSizes.size())
      return failure();

    for (auto [idx, staticSize] : llvm::enumerate(allStaticSizes)) {
      if (staticSize == ShapedType::kDynamic || staticSize != 1)
        mixedSizes.push_back(allMixedSizes[idx]);
    }
  } else {
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    if (!sourceType)
      return failure();

    for (auto [idx, dim] : llvm::enumerate(resultType.getShape())) {
      if (ShapedType::isDynamic(dim)) {
        if (idx >= static_cast<size_t>(sourceType.getRank()))
          return failure();
        mixedSizes.push_back(memref::DimOp::create(rewriter, loc, source, idx)
                                 .getResult());
      } else {
        mixedSizes.push_back(rewriter.getIndexAttr(dim));
      }
    }
  }

  if (mixedSizes.size() != static_cast<size_t>(resultType.getRank()))
    return failure();

  dispatchIndexOpFoldResults(mixedSizes, dynSizes, staticSizes);
  return success();
}

struct ToTensorLowering
    : public OpRewritePattern<bufferization::ToTensorOp> {
  using OpRewritePattern<bufferization::ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType)
      return failure();

    SmallVector<Value, 4> dynSizes;
    SmallVector<int64_t, 4> staticSizes;
    if (failed(deriveTensorSizes(op.getLoc(), op.getBuffer(), resultType,
                                 rewriter, dynSizes, staticSizes)))
      return failure();

    auto newOp = loom::BufferizeToTensorOp::create(
        rewriter, op.getLoc(), resultType, op.getBuffer(), dynSizes,
        rewriter.getDenseI64ArrayAttr(staticSizes));
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ToBufferLowering
    : public OpRewritePattern<bufferization::ToBufferOp> {
  using OpRewritePattern<bufferization::ToBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToBufferOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!resultType)
      return failure();

    auto newOp = loom::BufferizeToMemrefOp::create(
        rewriter, op.getLoc(), resultType, op.getTensor());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct CanonicalBufferizationToLoomPass
    : public PassWrapper<CanonicalBufferizationToLoomPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CanonicalBufferizationToLoomPass)

  StringRef getArgument() const override {
    return "loom-canonical-bufferization-to-loom";
  }

  StringRef getDescription() const override {
    return "Rewrite canonical bufferization handoff ops to loom dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect,
                    memref::MemRefDialect, loom::LoomDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ToTensorLowering, ToBufferLowering>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createCanonicalBufferizationToLoomPass() {
  return std::make_unique<CanonicalBufferizationToLoomPass>();
}
