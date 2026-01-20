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
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;
using namespace tt::ttcore;

namespace {

/**
 * @brief Tracks input memrefs and assigns compile-arg indices for CB/base addr.
 *
 * @details Each input memref is assigned a pair of compile-arg indices:
 *          - cbIndex: index for the L1 circular buffer.
 *          - baseAddrIndex: index for the input base address (cbIndex + 1).
 */
class InputDataTracker {
public:
  /**
   * @brief Encapsulates compile-arg indices for an input memref.
   */
  struct InputData {
    int64_t cbIndex;
    int64_t baseAddrIndex;
  };

  /**
   * @brief Get or create a CB index for the given L1 alloc.
   * @param alloc The `memref.alloc` op annotated with `{loom.alloc ...}`.
   * @return A stable, non-negative compile-arg index for this allocation.
   */
  int64_t getOrCreateAlloc(memref::AllocOp alloc) {
    Value v = alloc.getResult();
    auto it = allocToCbIndex.find(v);
    if (it != allocToCbIndex.end())
      return it->second;
    int64_t cbIndex = nextCompileArgIndex;
    nextCompileArgIndex += 2;
    allocToCbIndex.try_emplace(v, cbIndex);
    return cbIndex;
  }

  /**
   * @brief Get or create input data for the given input memref.
   * @param inputMemref Input memref value (typically a function argument).
   * @param alloc The L1 alloc associated with this input.
   * @return InputData containing CB index and base address index.
   */
  const InputData &getOrCreate(Value inputMemref, memref::AllocOp alloc) {
    auto it = inputToData.find(inputMemref);
    if (it != inputToData.end())
      return it->second;
    int64_t cbIndex = getOrCreateAlloc(alloc);
    int64_t baseAddrIndex = cbIndex + 1;
    InputData data{cbIndex, baseAddrIndex};
    auto inserted = inputToData.try_emplace(inputMemref, data);
    return inserted.first->second;
  }

private:
  /// Map from input memref to its compile-arg indices.
  llvm::DenseMap<Value, InputData> inputToData;
  /// Map from alloc result to its assigned CB index.
  llvm::DenseMap<Value, int64_t> allocToCbIndex;
  /// Next compile-arg index (even indices are CBs; odd are base addrs).
  int64_t nextCompileArgIndex = 0;
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
  ConvertLoadOp(TypeConverter &typeConverter, MLIRContext *context,
                std::shared_ptr<InputDataTracker> inputTracker)
      : OpConversionPattern<memref::CopyOp>(typeConverter, context),
        inputTracker(std::move(inputTracker)) {}

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
    Location loc = op.getLoc();
    Value cb;
    memref::AllocOp targetAlloc = op.getTarget().getDefiningOp<memref::AllocOp>();
    if (targetAlloc && targetAlloc->hasAttr("loom.alloc")) {
      int64_t allocIdx = inputTracker->getOrCreateAlloc(targetAlloc);
      auto idxAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(allocIdx));
      auto memrefType =
          cast<CBType>(typeConverter->convertType(op.getTarget().getType()));
      cb = rewriter.create<GetCompileArgValOp>(loc, memrefType, idxAttr);
    } else {
      cb = rewriter.getRemappedValue(op.getTarget());
    }

    auto opInsertionPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(cb);

    auto dataFormat = GetDataFormatOp::create(rewriter, loc, cb);
    auto pageSize = GetTileSizeOp::create(rewriter, loc, cb);

    // Get the base address and offset from memref.reinterpret_cast
    Value source = op.getSource();
    Value offset;
    Value inputMemref = source;
    if (auto reinterpretCastOp = source.getDefiningOp<memref::ReinterpretCastOp>()) {
      inputMemref = reinterpretCastOp.getSource();
      // Get the offset from reinterpret_cast (use first offset if multiple)
      auto offsets = reinterpretCastOp.getOffsets();
      if (!offsets.empty()) {
        offset = offsets[0];
      } else {
        // If no dynamic offsets, check static offsets
        auto mixedOffsets = reinterpretCastOp.getMixedOffsets();
        if (!mixedOffsets.empty() && mixedOffsets[0].is<Attribute>()) {
          // Static offset - convert to value
          auto staticOffset = llvm::cast<IntegerAttr>(mixedOffsets[0].get<Attribute>());
          offset = rewriter.create<arith::ConstantIndexOp>(loc, staticOffset.getInt());
        } else {
          // Fallback: use constant 0 (index type)
          offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        }
      }
    }

    // If the source comes from a memref input and the target is an L1 alloc,
    // materialize the base address from a compile-time arg adjacent to the CB.
        const auto &inputData =
            inputTracker->getOrCreate(inputMemref, targetAlloc);
        auto baseAddrIdxAttr = rewriter.getI32IntegerAttr(
            static_cast<int32_t>(inputData.baseAddrIndex));
        auto baseAddr = rewriter.create<GetCompileArgValOp>(
            loc, rewriter.getI32Type(), baseAddrIdxAttr);

    Value trueVal = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI1Type(), 1);
    Value addrGen = GetInterleavedAddrGenFastOp::create(
        rewriter, loc, /*dram=*/trueVal, baseAddr, pageSize, dataFormat);

    rewriter.restoreInsertionPoint(opInsertionPt);

    // convert bytes offset to tile index
    //
    // arith.divui requires operands/results to have the same type.
    // Our `offset` is typically `index` (from memref.reinterpret_cast), while
    // TTKernel tile size is `i32`. Convert offset to `i32` before dividing.
    Value offsetI32 = offset;
    if (offsetI32 && offsetI32.getType().isIndex()) {
      offsetI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                                      offsetI32);
    }
    Value baseTileIndex =
        arith::DivUIOp::create(rewriter, loc, offsetI32, pageSize);

    Value const0 = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), 0);

    // determine how many tiles we need to load by converting the shape to tiles
    llvm::outs() << "cb type: " << cast<CBType>(cb.getType()) << "\n";
    auto cbType = cast<CBType>(cb.getType());
    Value numPages;
    
    // Check if CB is tiled (element type is TileType) or scalar
    auto elementType = cbType.getElementType();
    if (auto tileType = llvm::dyn_cast<TileType>(elementType)) {
      // Tiled CB: use getNumTiles() directly
      const int32_t numTiles = cbType.getNumTiles();
      numPages = rewriter.create<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), numTiles);
    } else {
      // Scalar CB: calculate numTiles = (numElements * elementSizeInBytes) / pageSizeInBytes
      // Get number of elements and element size
      const int64_t numElements = cbType.getNumElements();
      
      // Get element size in bytes
      int32_t elementSizeBytes = 0;
      if (elementType.isF32()) {
        elementSizeBytes = 4;
      } else if (elementType.isF16() || elementType.isBF16()) {
        elementSizeBytes = 2;
      } else if (auto intType = llvm::dyn_cast<IntegerType>(elementType)) {
        elementSizeBytes = (intType.getWidth() + 7) / 8; // Round up to bytes
      } else {
        // Default: try to infer from type
        // For now, assume 4 bytes if unknown
        elementSizeBytes = 4;
      }
      
      // Calculate total size in bytes and divide by page size
      Value totalSizeBytes = rewriter.create<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), numElements * elementSizeBytes);
      numPages = arith::DivUIOp::create(rewriter, loc, totalSizeBytes, pageSize);
    }
    CBReserveBackOp::create(rewriter, loc, cb, numPages);

    Value l1Addr = GetWritePtrOp::create(rewriter, loc, cb);

    scf::ForOp loadTileLoop = scf::ForOp::create(
        rewriter, loc, rewriter.create<arith::ConstantIntOp>(
                          loc, rewriter.getI32Type(), 0),
        numPages,
        rewriter.create<arith::ConstantIntOp>(
            loc, rewriter.getI32Type(), 1),
        ValueRange{l1Addr, baseTileIndex});
    {
      rewriter.setInsertionPointToStart(loadTileLoop.getBody());
      Value crtL1Address = loadTileLoop.getRegionIterArgs()[0];
      Value crtTileIndex = loadTileLoop.getRegionIterArgs()[1];
      Value nocAddr = InterleavedAddrGenFastGetNocAddrOp::create(
          rewriter, loc, addrGen, crtTileIndex, const0, Value());
      NocAsyncReadOp::create(rewriter, loc, nocAddr, crtL1Address,
                                       pageSize);
      Value nextL1Address =
          arith::AddIOp::create(rewriter, loc, crtL1Address, pageSize);
      Value nextTileIndex =
          arith::AddIOp::create(rewriter, loc, crtTileIndex,
                                rewriter.create<arith::ConstantIntOp>(
                                    loc, rewriter.getI32Type(), 1));
      scf::YieldOp::create(rewriter, loc,
                           ValueRange{nextL1Address, nextTileIndex});
    }

    rewriter.setInsertionPointAfter(loadTileLoop);
    NocAsyncReadBarrierOp::create(rewriter, loc);


    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Shared input tracker for base address indices.
  std::shared_ptr<InputDataTracker> inputTracker;
};

/**
 * @brief Placeholder conversion for `memref.reinterpret_cast` annotated with
 *        `loom.reuse`.
 *
 * @details This is a framework hook to later lower reuse information into
 *          TTKernel-specific metadata/ops. For now, we simply recreate the
 *          `memref.reinterpret_cast` and drop the `loom.reuse` attribute so the
 *          operation becomes legal under the conversion target.
 */
struct ConvertReuseReinterpretCastOp
    : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle casts carrying the loom.reuse attribute.
    auto reuseAttr = op->getAttrOfType<DictionaryAttr>("loom.reuse");
    if (!reuseAttr)
      return failure();

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
                 std::shared_ptr<InputDataTracker> tracker)
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

/*     Location loc = op.getLoc();
    // Assign a stable index based on the number of tracked allocations so far.
    int64_t allocIdx = tracker->getOrCreateAlloc(op);
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
    rewriter.replaceOp(op, cb); */
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Shared input tracker used to assign deterministic indices.
  std::shared_ptr<InputDataTracker> tracker;
};

void loom::populateMemoryOpConversionPatterns(RewritePatternSet &patterns,
                                             TypeConverter &typeConverter,
                                             MLIRContext *context) {
  auto inputTracker = std::make_shared<InputDataTracker>();
  patterns.add<ConvertAllocOp>(typeConverter, context, inputTracker);
  patterns.add<ConvertReuseReinterpretCastOp>(typeConverter, context);
  patterns.add<ConvertLoadOp>(typeConverter, context, inputTracker);
}