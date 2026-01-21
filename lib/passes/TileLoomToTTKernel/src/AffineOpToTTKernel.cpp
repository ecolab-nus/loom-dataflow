/**
 * @file AffineOpToTTKernel.cpp
 * @brief Conversion of Affine operations (notably affine.parallel) to TTKernel.
 *
 * @details
 * This file provides patterns that remove `affine.parallel` loops and replace
 * their induction variables with TTKernel compile-time arguments, similar to
 * the way `CompileArgTracker` handles function arguments.
 *
 * For each `affine.parallel` operation without reductions:
 * - A fresh compile-arg index is allocated for each induction variable using
 *   `CompileArgTracker::getOrCreateIndex`.
 * - A `ttkernel.get_compile_time_arg_val` is emitted (returning `i32`), which
 *   is then cast back to `index` via `arith.index_cast`.
 * - All uses of the induction variable are replaced with this cast value.
 * - The loop body is inlined into the parent block and the `affine.parallel`
 *   op is erased.
 *
 * This treats the parallel iterators as per-kernel compile-time constants
 * rather than dynamic loop indices, which matches the intended mapping of
 * TileLoom spatial loops to hardware coordinates.
 */

#include "AffineOpToTTKernel.h"

#include "FuncOpToTTKernel.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;

namespace {

/**
 * @brief Convert `affine.parallel` to straight-line code with compile-time IVs.
 *
 * @details
 * This pattern targets `affine.parallel` operations with no reductions or
 * results. For each induction variable:
 *
 * - A fresh compile-time argument index is allocated via
 *   `CompileArgTracker::getOrCreateIndex`, using the IV value as the key.
 * - A `ttkernel.get_compile_time_arg_val` is emitted (type `i32`), followed
 *   by an `arith.index_cast` to turn it into an `index`-typed SSA value.
 * - All uses of the induction variable inside the loop body are rewritten to
 *   use this cast value instead.
 *
 * After induction variables are rewritten, the body of the `affine.parallel`
 * operation is inlined into the parent block and the original loop op is
 * erased. The original lower/upper bounds and steps are ignored, as the
 * parallel iterator is interpreted as a hardware coordinate provided as a
 * compile-time argument.
 */
class ConvertAffineParallelOp
    : public OpConversionPattern<affine::AffineParallelOp> {
public:
  using OpConversionPattern<affine::AffineParallelOp>::OpConversionPattern;

  ConvertAffineParallelOp(TypeConverter &typeConverter, MLIRContext *context,
                          std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<affine::AffineParallelOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  LogicalResult
  matchAndRewrite(affine::AffineParallelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Do not attempt to convert parallel loops with reductions/results for now.
    if (!op.getResults().empty())
      return failure();

    Location loc = op.getLoc();

    // For each induction variable, create a compile-time argument and cast it
    // back to index. We materialize the compile-arg loads just before the
    // affine.parallel op so they are in the surrounding function body.
    SmallVector<Value, 4> newIvValues;
    newIvValues.reserve(op.getIVs().size());

    rewriter.setInsertionPoint(op);
    for (Value iv : op.getIVs()) {
      // Allocate or look up a compile-arg index for this IV.
      int64_t argIndex = tracker->getOrCreateIndex(iv, op);
      auto idxAttr =
          rewriter.getI32IntegerAttr(static_cast<int32_t>(argIndex));

      // Emit TTKernel get_compile_time_arg_val (returns i32).
      Value ctArg =
          rewriter.create<GetCompileArgValOp>(loc, rewriter.getI32Type(),
                                              idxAttr);

      // Cast back to index so existing affine/arithmetic uses remain valid.
      Value ivIndex = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), ctArg);

      newIvValues.push_back(ivIndex);
    }

    // Replace all IV uses in the loop body with the new compile-arg-based
    // index values.
    for (auto [oldIv, newIv] : llvm::zip(op.getIVs(), newIvValues)) {
      oldIv.replaceAllUsesWith(newIv);
    }

    // Inline the body of the affine.parallel into the parent block, skipping
    // the terminator (affine.yield).
    Block *bodyBlock = op.getBody();

    // Move all operations except the terminator before the parallel op.
    for (Operation &innerOp :
         llvm::make_early_inc_range(bodyBlock->getOperations())) {
      if (isa<affine::AffineYieldOp>(innerOp))
        continue;
      innerOp.moveBefore(op);
    }

    // Erase the now-empty affine.parallel operation.
    rewriter.eraseOp(op);

    return success();
  }

private:
  /// Shared tracker for compile-arg index assignment.
  std::shared_ptr<CompileArgTracker> tracker;
};

} // namespace

void loom::populateAffineOpConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context, std::shared_ptr<CompileArgTracker> tracker) {
  patterns.add<ConvertAffineParallelOp>(typeConverter, context,
                                        std::move(tracker));
}


