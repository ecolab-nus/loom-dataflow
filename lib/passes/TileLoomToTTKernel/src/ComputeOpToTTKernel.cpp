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

    // For now, assume a single tile/block at index 0 for inputs and output.
    Value zeroI32 = rewriter.create<arith::ConstantIntOp>(
        loc, /*value=*/0, /*width=*/32);
    // Placeholder dimensions for a single CT/RT/KT/NT block.
    Value in0TileIdx = zeroI32;
    Value in1TileIdx = zeroI32;
    Value dstTileIdx = zeroI32;
    Value transpose = zeroI32;
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
    Value ctDim = rewriter.create<arith::ConstantIntOp>(loc, ctVal, 32);
    Value rtDim = rewriter.create<arith::ConstantIntOp>(loc, rtVal, 32);
    //TODO: ntDim should the number of tiles in final output N dimension
    Value ntDim = rewriter.create<arith::ConstantIntOp>(loc, ntVal, 32);
    Value ktDim = rewriter.create<arith::ConstantIntOp>(loc, ktVal, 32);


    // Init matmul block at the beginning of the function (after CB definitions)
    // and acquire tile registers once per kernel, before the main matmul body.
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (parentFunc) {
      auto &entryBlock = parentFunc.getBody().front();
      auto savedInsertionPt = rewriter.saveInsertionPoint();
      
      // Find insertion point after all CB definitions
      // CBs are defined by GetArgValOp or similar ops at function start
      Operation *insertAfter = nullptr;
      for (Operation &entryOp : entryBlock) {
        if (entryOp.getNumResults() > 0 && 
            isa<CBType>(entryOp.getResult(0).getType())) {
          insertAfter = &entryOp;
        }
      }
      
      if (insertAfter) {
        rewriter.setInsertionPointAfter(insertAfter);
      } else {
        rewriter.setInsertionPointToStart(&entryBlock);
      }
      
      rewriter.create<MatmulBlockInitOp>(
          loc, TypeRange{},
          ValueRange{in0Cb, in1Cb, outCb, transpose,
                     ctDim, rtDim, ktDim});

      rewriter.restoreInsertionPoint(savedInsertionPt);
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

} // namespace

void loom::populateComputeOpConversionPatterns(RewritePatternSet &patterns,
                                               TypeConverter &typeConverter,
                                               MLIRContext *context) {
  patterns.add<ConvertLinalgMatmulOp>(typeConverter, context);
}
