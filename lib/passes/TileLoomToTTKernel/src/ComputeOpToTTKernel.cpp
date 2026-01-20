/**
 * @file ComputeOpToTTKernel.cpp
 * @brief Skeleton conversions for compute ops (e.g., linalg.matmul) to TTKernel.
 *
 * @details Core lowering logic intentionally left empty; this file provides the
 *          pattern scaffolding so future implementations can plug in TTKernel
 *          compute lowering.
 */

#include "ComputeOpToTTKernel.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

/**
 * @brief Placeholder conversion for `linalg.matmul` to TTKernel.
 *
 * @details The match succeeds, but the rewrite is intentionally empty; replace
 *          with TTKernel-specific lowering when available.
 */
class ConvertLinalgMatmulOp : public OpConversionPattern<linalg::MatmulOp> {
public:
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  /**
   * @brief Match and rewrite `linalg.matmul`.
   *
   * @details Core lowering is not implemented; this is a scaffold to be filled
   *          with TTKernel compute lowering.
   */
  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Implement lowering to TTKernel compute ops.
    return failure();
  }
};

} // namespace

void loom::populateComputeOpConversionPatterns(RewritePatternSet &patterns,
                                               TypeConverter &typeConverter,
                                               MLIRContext *context) {
  patterns.add<ConvertLinalgMatmulOp>(typeConverter, context);
}
