/**
 * @file MemoryOpToTTKernel.cpp
 * @brief Implementation for memory operation to TT kernel conversion pass.
 * @details
 * This pass processes memory operations whose destination allocations carry
 * `{loom.alloc ...}` attributes and records their base address information
 * using the DataLoaderInfo structure.
 */

#include "MemoryOpToTTKernel.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;

namespace {

/**
 * @brief Tracks `loom.alloc` allocations and assigns a stable, dense index.
 *
 * @details The TTKernel lowering uses an integer "compile-arg index" to refer
 *          to circular buffers. When the input IR contains `memref.alloc` ops
 *          annotated with `{loom.alloc ...}`, we assign each such allocation
 *          a stable index based on insertion order into this tracker.
 *
 *          This avoids trying to infer indices from users (which fails when an
 *          alloc is currently unused) and makes lowering deterministic.
 */
class AllocTracker {
public:
  /**
   * @brief Get the index for an allocation, creating one if needed.
   * @param alloc The `memref.alloc` op annotated with `{loom.alloc ...}`.
   * @return A stable, non-negative index for this allocation.
   */
  int64_t getOrCreate(memref::AllocOp alloc) {
    Value v = alloc.getResult();
    auto it = allocToIndex.find(v);
    if (it != allocToIndex.end())
      return it->second;
    int64_t idx = static_cast<int64_t>(allocs.size());
    allocs.push_back(v);
    allocToIndex.try_emplace(v, idx);
    return idx;
  }

private:
  /// Ordered list of tracked alloc results.
  llvm::SmallVector<Value, 8> allocs;
  /// Map from alloc result to its assigned index.
  llvm::DenseMap<Value, int64_t> allocToIndex;
};

} // namespace
/**
 * @brief Convert `memref.copy` with `loom.copy.choice` into TTKernel load ops.
 *
 * @details The conversion uses `DataLoaderInfo` to recover the base memref and
 *          offset from the source `memref.reinterpret_cast`. It then emits a
 *          TTKernel NOC read sequence to populate the destination circular
 *          buffer (CB). The destination is expected to be type-converted to a
 *          TTKernel CB type by the surrounding conversion pipeline.
 */
struct ConvertLoadOp : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  /**
   * @brief Construct the pattern with a type converter and context.
   * @param typeConverter Type converter for the conversion pipeline.
   * @param context MLIR context.
   */
  ConvertLoadOp(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<memref::CopyOp>(typeConverter, context) {}

  /**
   * @brief Match and rewrite `memref.copy` into a no-op for TTKernel lowering.
   *
   * @details For now, we implement a simple version that **removes**
   *          `memref.copy` operations whose destination allocation is
   *          annotated with `{loom.alloc ...}`. This satisfies the conversion
   *          target without yet introducing TTKernel-specific load operations.
   */
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle memref.copy with the loom.copy.choice attribute.
    auto copyChoiceAttr =
        op->getAttrOfType<DictionaryAttr>("loom.copy.choice");
    if (!copyChoiceAttr)
      return failure();

    // Optionally check the kind field (default to "mem" if present).
    if (auto kindAttr = copyChoiceAttr.getAs<StringAttr>("kind")) {
      if (kindAttr.getValue() != "mem")
        return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};


struct ConvertAllocOp : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  /**
   * @brief Construct the pattern with a type converter and context.
   * @param typeConverter Type converter for the conversion pipeline.
   * @param context MLIR context.
   */
  ConvertAllocOp(TypeConverter &typeConverter, MLIRContext *context,
                 std::shared_ptr<AllocTracker> tracker)
      : OpConversionPattern<memref::AllocOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  /**
   * @brief Match and rewrite `memref.alloc` with `{loom.alloc ...}`.
   *
   * @details This pattern is intended to recognize allocations that are meant
   *          to become TTKernel circular buffers (CBs) later in the pipeline.
   *          For now, it is a no-op and only serves as a hook point for future
   *          lowering.
   */
  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle memref.alloc annotated with {loom.alloc ...}.
    auto allocAttr = op->getAttrOfType<DictionaryAttr>("loom.alloc");
    if (!allocAttr)
      return failure();

    Location loc = op.getLoc();
    // Assign a stable index based on the number of tracked allocations so far.
    int64_t allocIdx = tracker->getOrCreate(op);
    auto idxAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(allocIdx));
    auto memrefType =
        cast<CBType>(typeConverter->convertType(op.getResult().getType()));
    auto cb =
        rewriter.create<GetCompileArgValOp>(loc, memrefType, idxAttr);
    //auto opInsertionPt = rewriter.saveInsertionPoint();
    //rewriter.setInsertionPointAfterValue(cb);

    //auto dataFormat = GetDataFormatOp::create(rewriter, loc, cb);
    //auto pageSize = GetTileSizeOp::create(rewriter, loc, cb);

    // Replace all uses of the alloc result with the CB value
    rewriter.replaceOp(op, cb);
    //rewriter.eraseOp(op);
    return success();
  }

private:
  /// Shared allocation tracker used to assign deterministic indices.
  std::shared_ptr<AllocTracker> tracker;
};

void loom::populateMemoryOpConversionPatterns(RewritePatternSet &patterns,
                                             TypeConverter &typeConverter,
                                             MLIRContext *context) {
  auto tracker = std::make_shared<AllocTracker>();
  patterns.add<ConvertAllocOp>(typeConverter, context, tracker);
  patterns.add<ConvertLoadOp>(typeConverter, context);
}