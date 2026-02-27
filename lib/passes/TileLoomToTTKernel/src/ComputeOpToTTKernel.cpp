/**
 * @file ComputeOpToTTKernel.cpp
 * @brief Skeleton conversions for compute ops (e.g., linalg.matmul) to TTKernel.
 *
 * @details Core lowering logic for TTKernel matmul is minimal and currently
 *          emits a single `ttkernel.experimental::matmul_block` op with
 *          placeholder tiling parameters. This is sufficient to exercise the
 *          conversion on simple test cases; more advanced tiling and init
 *          sequencing can be added later.
 */

#include "ComputeOpToTTKernel.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include <limits>

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

using namespace mlir;
using namespace tt::ttkernel;

namespace {

class ConvertLinalgMatmulOp : public OpConversionPattern<linalg::MatmulOp> {
public:
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  /**
   * @brief Match and rewrite `linalg.matmul`.
   *
   * @details This initial lowering maps a rank-2 `linalg.matmul` with CB-typed
   *          operands to a single `ttkernel.experimental::matmul_block` op.
   *          Tiling parameters are currently hard-coded for simple test cases
   *          and should be generalized in future revisions.
   */
  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expect exactly two inputs and one output.
    if (adaptor.getInputs().size() != 2 || adaptor.getOutputs().size() != 1)
      return failure();

    Location loc = op.getLoc();

    Value in0Cb = adaptor.getInputs()[0];
    Value in1Cb = adaptor.getInputs()[1];
    Value outCb = adaptor.getOutputs()[0];

    // Ensure operands are TTKernel CBs.
    if (!isa<CBType>(in0Cb.getType()) || !isa<CBType>(in1Cb.getType()) ||
        !isa<CBType>(outCb.getType()))
      return failure();

    // Use the original linalg.matmul input type to query the tensor/memref shape.
    auto lhsShapedType = dyn_cast<ShapedType>(op.getInputs()[0].getType());
    ArrayRef<int64_t> lhsShape = lhsShapedType.getShape();
    int32_t rtVal = static_cast<int32_t>(lhsShape[0] / 32);
    int32_t ktVal = static_cast<int32_t>(lhsShape[1] / 32);
    auto rhsShapedType = dyn_cast<ShapedType>(op.getInputs()[1].getType());
    ArrayRef<int64_t> rhsShape = rhsShapedType.getShape();
    int32_t ctVal = static_cast<int32_t>(rhsShape[1] / 32);
    auto outShapedType = dyn_cast<ShapedType>(op.getOutputs()[0].getType());
    ArrayRef<int64_t> outShape = outShapedType.getShape();
    int32_t ntVal = static_cast<int32_t>(outShape[1] / 32);
    Value zeroI32;
    Value in0TileIdx;
    Value in1TileIdx;
    Value dstTileIdx;
    Value transpose;
    Value ctDim;
    Value rtDim;
    Value ntDim;
    Value ktDim;

    {
      OpBuilder::InsertionGuard guard(rewriter);
      bool placeInitAtKernelStart = false;

      if (auto parentFunc = op->getParentOfType<func::FuncOp>()) {
        Block &entry = parentFunc.front();
        Operation *lastGetArgValOp = nullptr;
        for (Operation &entryOp : entry)
          if (isa<GetArgValOp>(entryOp))
            lastGetArgValOp = &entryOp;

        auto isAvailableAtEntry = [&](Value value) -> bool {
          if (auto blockArg = dyn_cast<BlockArgument>(value))
            return blockArg.getOwner() == &entry;

          Operation *defOp = value.getDefiningOp();
          if (!defOp || defOp->getBlock() != &entry)
            return false;

          if (!lastGetArgValOp)
            return true;

          return defOp == lastGetArgValOp ||
                 defOp->isBeforeInBlock(lastGetArgValOp);
        };

        placeInitAtKernelStart = isAvailableAtEntry(in0Cb) &&
                                 isAvailableAtEntry(in1Cb) &&
                                 isAvailableAtEntry(outCb);

        if (placeInitAtKernelStart) {
          if (lastGetArgValOp)
            rewriter.setInsertionPointAfter(lastGetArgValOp);
          else
            rewriter.setInsertionPointToStart(&entry);
        }
      }

      zeroI32 = rewriter.create<arith::ConstantIntOp>(
          loc, /*value=*/0, /*width=*/32);
      in0TileIdx = zeroI32;
      in1TileIdx = zeroI32;
      dstTileIdx = zeroI32;
      transpose = zeroI32;
      ctDim = rewriter.create<arith::ConstantIntOp>(loc, ctVal, 32);
      rtDim = rewriter.create<arith::ConstantIntOp>(loc, rtVal, 32);
      ntDim = rewriter.create<arith::ConstantIntOp>(loc, ntVal, 32);
      ktDim = rewriter.create<arith::ConstantIntOp>(loc, ktVal, 32);

      rewriter.create<MatmulBlockInitOp>(
          loc, TypeRange{},
          ValueRange{in0Cb, in1Cb, outCb, transpose, ctDim, rtDim, ktDim});
    }
    // cb wait front - get number of tiles/pages for each CB
    auto in0CbType = cast<CBType>(in0Cb.getType());
    auto in1CbType = cast<CBType>(in1Cb.getType());

    // Use getNumElements() which works for both tiled and scalar CBs
    // (getNumTiles() just calls getNumElements() but asserts element type is TileType)
    //TODO: this should be a tiled CB, now is using elements
    Value in0NumInputTilesValue = rewriter.create<arith::ConstantIntOp>(
        loc, static_cast<int32_t>(in0CbType.getNumElements()) / 1024, 32);
    Value in1NumInputTilesValue = rewriter.create<arith::ConstantIntOp>(
        loc, static_cast<int32_t>(in1CbType.getNumElements()) / 1024, 32);
    CBWaitFrontOp::create(rewriter, loc, in0Cb, in0NumInputTilesValue);
    CBWaitFrontOp::create(rewriter, loc, in1Cb, in1NumInputTilesValue);

    // Emit experimental::matmul_block. Core semantics and tuning are deferred.
    rewriter.create<ExperimentalMatmulBlockOp>(
        loc, TypeRange{},
        ValueRange{in0Cb, in1Cb, in0TileIdx, in1TileIdx, dstTileIdx, transpose,
                   ctDim, rtDim, ktDim, ntDim});
    
    //pop input cb
    CBPopFrontOp::create(rewriter, loc, in0Cb, in0NumInputTilesValue);
    CBPopFrontOp::create(rewriter, loc, in1Cb, in1NumInputTilesValue);

    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertLinalgBatchMatmulOp
    : public OpConversionPattern<linalg::BatchMatmulOp> {
public:
  using OpConversionPattern<linalg::BatchMatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::BatchMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expect exactly two inputs and one output.
    if (adaptor.getInputs().size() != 2 || adaptor.getOutputs().size() != 1)
      return failure();

    Location loc = op.getLoc();

    Value in0Cb = adaptor.getInputs()[0];
    Value in1Cb = adaptor.getInputs()[1];
    Value outCb = adaptor.getOutputs()[0];

    // Ensure operands are TTKernel CBs.
    if (!isa<CBType>(in0Cb.getType()) || !isa<CBType>(in1Cb.getType()) ||
        !isa<CBType>(outCb.getType()))
      return failure();

    auto lhsShapedType = dyn_cast<ShapedType>(op.getInputs()[0].getType());
    auto rhsShapedType = dyn_cast<ShapedType>(op.getInputs()[1].getType());
    auto outShapedType = dyn_cast<ShapedType>(op.getOutputs()[0].getType());
    if (!lhsShapedType || !rhsShapedType || !outShapedType)
      return failure();
    if (!lhsShapedType.hasStaticShape() || !rhsShapedType.hasStaticShape() ||
        !outShapedType.hasStaticShape())
      return failure();
    if (lhsShapedType.getRank() != 3 || rhsShapedType.getRank() != 3 ||
        outShapedType.getRank() != 3)
      return failure();

    ArrayRef<int64_t> lhsShape = lhsShapedType.getShape(); // [B, M, K]
    ArrayRef<int64_t> rhsShape = rhsShapedType.getShape(); // [B, K, N]
    ArrayRef<int64_t> outShape = outShapedType.getShape(); // [B, M, N]

    int64_t batchVal64 = outShape[0];
    int64_t mVal64 = lhsShape[1];
    int64_t kVal64 = lhsShape[2];
    int64_t nVal64 = outShape[2];

    // Syntax-level shape sanity checks for canonical batch matmul intent:
    // [B, M, K] x [B, K, N] -> [B, M, N].
    if (rhsShape[0] != batchVal64 || rhsShape[1] != kVal64 ||
        rhsShape[2] != nVal64 || outShape[1] != mVal64)
      return failure();

    if (batchVal64 <= 0 || mVal64 <= 0 || kVal64 <= 0 || nVal64 <= 0)
      return failure();
    if (mVal64 % 32 != 0 || kVal64 % 32 != 0 || nVal64 % 32 != 0)
      return failure();

    int32_t batchVal = static_cast<int32_t>(batchVal64);
    int32_t rtVal = static_cast<int32_t>(mVal64 / 32);
    int32_t ktVal = static_cast<int32_t>(kVal64 / 32);
    int32_t ctVal = static_cast<int32_t>(nVal64 / 32);
    int32_t ntVal = static_cast<int32_t>(nVal64 / 32);

    int64_t in0TilesPerBatch64 = static_cast<int64_t>(rtVal) * ktVal;
    int64_t in1TilesPerBatch64 = static_cast<int64_t>(ktVal) * ctVal;
    int64_t outTilesPerBatch64 = static_cast<int64_t>(rtVal) * ntVal;
    if (in0TilesPerBatch64 > std::numeric_limits<int32_t>::max() ||
        in1TilesPerBatch64 > std::numeric_limits<int32_t>::max() ||
        outTilesPerBatch64 > std::numeric_limits<int32_t>::max())
      return failure();

    Value zeroI32;
    Value transpose;
    Value ctDim;
    Value rtDim;
    Value ntDim;
    Value ktDim;

    {
      OpBuilder::InsertionGuard guard(rewriter);
      bool placeInitAtKernelStart = false;

      if (auto parentFunc = op->getParentOfType<func::FuncOp>()) {
        Block &entry = parentFunc.front();
        Operation *lastGetArgValOp = nullptr;
        for (Operation &entryOp : entry)
          if (isa<GetArgValOp>(entryOp))
            lastGetArgValOp = &entryOp;

        auto isAvailableAtEntry = [&](Value value) -> bool {
          if (auto blockArg = dyn_cast<BlockArgument>(value))
            return blockArg.getOwner() == &entry;

          Operation *defOp = value.getDefiningOp();
          if (!defOp || defOp->getBlock() != &entry)
            return false;

          if (!lastGetArgValOp)
            return true;

          return defOp == lastGetArgValOp ||
                 defOp->isBeforeInBlock(lastGetArgValOp);
        };

        placeInitAtKernelStart = isAvailableAtEntry(in0Cb) &&
                                 isAvailableAtEntry(in1Cb) &&
                                 isAvailableAtEntry(outCb);

        if (placeInitAtKernelStart) {
          if (lastGetArgValOp)
            rewriter.setInsertionPointAfter(lastGetArgValOp);
          else
            rewriter.setInsertionPointToStart(&entry);
        }
      }

      zeroI32 = rewriter.create<arith::ConstantIntOp>(
          loc, /*value=*/0, /*width=*/32);
      transpose = zeroI32;
      ctDim = rewriter.create<arith::ConstantIntOp>(loc, ctVal, 32);
      rtDim = rewriter.create<arith::ConstantIntOp>(loc, rtVal, 32);
      ntDim = rewriter.create<arith::ConstantIntOp>(loc, ntVal, 32);
      ktDim = rewriter.create<arith::ConstantIntOp>(loc, ktVal, 32);

      rewriter.create<MatmulBlockInitOp>(
          loc, TypeRange{},
          ValueRange{in0Cb, in1Cb, outCb, transpose, ctDim, rtDim, ktDim});
    }

    // cb wait front - get number of tiles/pages for each CB.
    auto in0CbType = cast<CBType>(in0Cb.getType());
    auto in1CbType = cast<CBType>(in1Cb.getType());

    Value in0NumInputTilesValue = rewriter.create<arith::ConstantIntOp>(
        loc, static_cast<int32_t>(in0CbType.getNumElements()) / 1024, 32);
    Value in1NumInputTilesValue = rewriter.create<arith::ConstantIntOp>(
        loc, static_cast<int32_t>(in1CbType.getNumElements()) / 1024, 32);
    CBWaitFrontOp::create(rewriter, loc, in0Cb, in0NumInputTilesValue);
    CBWaitFrontOp::create(rewriter, loc, in1Cb, in1NumInputTilesValue);

    for (int32_t b = 0; b < batchVal; ++b) {
      int64_t in0TileOffset64 = static_cast<int64_t>(b) * in0TilesPerBatch64;
      int64_t in1TileOffset64 = static_cast<int64_t>(b) * in1TilesPerBatch64;
      int64_t dstTileOffset64 = static_cast<int64_t>(b) * outTilesPerBatch64;
      if (in0TileOffset64 > std::numeric_limits<int32_t>::max() ||
          in1TileOffset64 > std::numeric_limits<int32_t>::max() ||
          dstTileOffset64 > std::numeric_limits<int32_t>::max())
        return failure();

      Value in0TileIdx = rewriter.create<arith::ConstantIntOp>(
          loc, static_cast<int32_t>(in0TileOffset64), 32);
      Value in1TileIdx = rewriter.create<arith::ConstantIntOp>(
          loc, static_cast<int32_t>(in1TileOffset64), 32);
      Value dstTileIdx = rewriter.create<arith::ConstantIntOp>(
          loc, static_cast<int32_t>(dstTileOffset64), 32);

      rewriter.create<ExperimentalMatmulBlockOp>(
          loc, TypeRange{},
          ValueRange{in0Cb, in1Cb, in0TileIdx, in1TileIdx, dstTileIdx,
                     transpose, ctDim, rtDim, ktDim, ntDim});
    }

    // pop input cb.
    CBPopFrontOp::create(rewriter, loc, in0Cb, in0NumInputTilesValue);
    CBPopFrontOp::create(rewriter, loc, in1Cb, in1NumInputTilesValue);

    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertLinalgFillOp : public OpConversionPattern<linalg::FillOp> {
public:
  using OpConversionPattern<linalg::FillOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void loom::populateComputeOpConversionPatterns(RewritePatternSet &patterns,
                                               TypeConverter &typeConverter,
                                               MLIRContext *context) {
  patterns.add<ConvertLinalgMatmulOp>(typeConverter, context);
  patterns.add<ConvertLinalgBatchMatmulOp>(typeConverter, context);
  patterns.add<ConvertLinalgFillOp>(typeConverter, context);
}
