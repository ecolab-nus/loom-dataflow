/**
 * @file ComputeOpToTTKernel.cpp
 * @brief Skeleton conversions for compute ops (e.g., linalg.matmul) to TTKernel.
 *
 * @details Core lowering logic intentionally left empty; this file provides the
 *          pattern scaffolding so future implementations can plug in TTKernel
 *          compute lowering.
 */

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
    Value oneI32 = rewriter.create<arith::ConstantIntOp>(
        loc, /*value=*/1, /*width=*/32);

    // Placeholder dimensions for a single CT/RT/KT/NT block.
    Value in0TileIdx = zeroI32;
    Value in1TileIdx = zeroI32;
    Value dstTileIdx = zeroI32;
    Value transpose = zeroI32;
    Value ctDim = oneI32;
    Value rtDim = oneI32;
    Value ktDim = oneI32;
    Value ntDim = oneI32;


    // init matmul block at the beginning of the function (after CB definitions)
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (parentFunc) {
      auto &entryBlock = parentFunc.getBody().front();
      auto savedInsertionPt = rewriter.saveInsertionPoint();
      
      // Find insertion point after all CB definitions
      // CBs are defined by GetCompileArgValOp or similar ops at function start
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

    //add lock for DST register
    TileRegsAcquireOp::create(rewriter, loc);
    
    // Use getNumElements() which works for both tiled and scalar CBs
    // (getNumTiles() just calls getNumElements() but asserts element type is TileType)
    Value in0NumInputTilesValue = rewriter.create<arith::ConstantIntOp>(
        loc, static_cast<int32_t>(in0CbType.getNumElements()), 32);
    Value in1NumInputTilesValue = rewriter.create<arith::ConstantIntOp>(
        loc, static_cast<int32_t>(in1CbType.getNumElements()), 32);
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
