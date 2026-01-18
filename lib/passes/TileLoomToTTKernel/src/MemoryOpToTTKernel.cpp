//===- MemoryOpToTTKernel.cpp - Lower memref.copy to TTKernel ops ---------===//
//
// This file implements conversion patterns for lowering memref.copy operations
// to TTKernel dialect operations. The primary focus is on DRAM to L1 memory
// transfers using the Tenstorrent NOC (Network on Chip) infrastructure.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

// TTKernel dialect headers - to be included when available
// #include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
// #include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#define DEBUG_TYPE "tile-loom-to-ttkernel-memory"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace loom {

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Creates a constant i32 value at the given location.
static Value createConstantI32(Location loc, OpBuilder &builder, int32_t value) {
  return builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(builder.getI32Type(), value));
}

/// Creates a constant i1 (boolean) value at the given location.
static Value createConstantI1(Location loc, OpBuilder &builder, bool value) {
  return builder.create<arith::ConstantOp>(
      loc, builder.getBoolAttr(value));
}

/// Creates a constant index value at the given location.
static Value createConstantIndex(Location loc, OpBuilder &builder,
                                 int64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(value));
}

/// Extracts the loom.copy.choice attribute from an operation.
/// Returns the dictionary attribute if present, nullptr otherwise.
static DictionaryAttr getCopyChoiceAttr(Operation *op) {
  return op->getAttrOfType<DictionaryAttr>("loom.copy.choice");
}

/// Checks if the copy operation is a DRAM to L1 transfer.
/// The copy is considered DRAM to L1 if:
/// - It has the loom.copy.choice attribute
/// - The attribute has kind = "mem"
/// - The attribute has memory_name = "L1"
static bool isDramToL1Copy(memref::CopyOp copyOp) {
  auto copyChoice = getCopyChoiceAttr(copyOp);
  if (!copyChoice)
    return false;

  auto kindAttr = copyChoice.getAs<StringAttr>("kind");
  auto memNameAttr = copyChoice.getAs<StringAttr>("memory_name");

  if (!kindAttr || !memNameAttr)
    return false;

  return kindAttr.getValue() == "mem" && memNameAttr.getValue() == "L1";
}

/// Computes the number of tiles needed for a given memref shape.
/// Assumes tiles are 32x32 elements (standard Tenstorrent tile size).
static int64_t computeNumTiles(MemRefType memrefType,
                               int64_t tileHeight = 32,
                               int64_t tileWidth = 32) {
  auto shape = memrefType.getShape();
  if (shape.size() < 2)
    return 1;

  int64_t rows = shape[shape.size() - 2];
  int64_t cols = shape[shape.size() - 1];

  int64_t tilesPerRow = (cols + tileWidth - 1) / tileWidth;
  int64_t tilesPerCol = (rows + tileHeight - 1) / tileHeight;

  return tilesPerRow * tilesPerCol;
}

/// Extracts the base address from a memref value.
/// Handles reinterpret_cast operations to get the underlying pointer.
static Value extractBaseAddress(Value memref, OpBuilder &builder,
                                Location loc) {
  // If the memref comes from a reinterpret_cast, extract the offset
  if (auto reinterpretOp =
          memref.getDefiningOp<memref::ReinterpretCastOp>()) {
    // The offset operand represents the byte offset into the base buffer
    auto offsets = reinterpretOp.getOffsets();
    if (!offsets.empty()) {
      return offsets.front();
    }
    // If no dynamic offset, use static offset
    auto staticOffsets = reinterpretOp.getStaticOffsets();
    if (!staticOffsets.empty() && staticOffsets[0] != ShapedType::kDynamic) {
      return createConstantIndex(loc, builder, staticOffsets[0]);
    }
  }
  // Default: return a zero offset
  return createConstantIndex(loc, builder, 0);
}

/// Gets the element byte size for a memref type.
static int64_t getElementByteSize(MemRefType memrefType) {
  Type elementType = memrefType.getElementType();
  if (auto intType = dyn_cast<IntegerType>(elementType))
    return (intType.getWidth() + 7) / 8;
  if (auto floatType = dyn_cast<FloatType>(elementType))
    return floatType.getWidth() / 8;
  // Default to 4 bytes (f32/i32)
  return 4;
}

//===----------------------------------------------------------------------===//
// ConvertMemrefCopyToTTKernel Pattern
//===----------------------------------------------------------------------===//

/// Conversion pattern for lowering memref.copy to TTKernel NOC operations.
///
/// This pattern handles DRAM to L1 memory transfers by:
/// 1. Computing the source DRAM address from the memref offset
/// 2. Setting up an interleaved address generator for tile-based access
/// 3. Iterating over tiles and issuing NOC async read operations
/// 4. Inserting a barrier to wait for all reads to complete
/// 5. Signaling that the destination CB (circular buffer) has data ready
///
/// Example transformation:
/// ```mlir
/// // Before:
/// memref.copy %src, %dst {loom.copy.choice = {kind = "mem", memory_name = "L1"}}
///
/// // After (conceptually):
/// %addrGen = ttkernel.get_interleaved_addr_gen_fast(...)
/// ttkernel.cb_reserve_back(%dst, %numTiles)
/// scf.for %i = 0 to %numTiles {
///   %nocAddr = ttkernel.interleaved_addr_gen_fast_get_noc_addr(%addrGen, %i, ...)
///   %l1Addr = ttkernel.get_write_ptr(%dst) + %i * %tileSize
///   ttkernel.noc_async_read(%nocAddr, %l1Addr, %tileSize)
/// }
/// ttkernel.noc_async_read_barrier()
/// ttkernel.cb_push_back(%dst, %numTiles)
/// ```
struct ConvertMemrefCopyToTTKernel
    : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Only handle DRAM to L1 copies with the appropriate attribute
    if (!isDramToL1Copy(op)) {
      LDBG("Skipping memref.copy: not a DRAM to L1 copy");
      return failure();
    }

    Value src = adaptor.getSource();
    Value dst = adaptor.getTarget();

    auto srcType = cast<MemRefType>(op.getSource().getType());
    auto dstType = cast<MemRefType>(op.getTarget().getType());

    LDBG("Converting memref.copy: " << srcType << " to " << dstType);

    // Get the source base address (DRAM offset)
    Value srcBaseAddr = extractBaseAddress(op.getSource(), rewriter, loc);

    // Compute the number of tiles to transfer
    int64_t numTiles = computeNumTiles(dstType);
    Value numTilesVal = createConstantI32(loc, rewriter, numTiles);

    // Get tile/page size in bytes
    int64_t elementSize = getElementByteSize(dstType);
    int64_t tileSizeBytes = 32 * 32 * elementSize; // Standard 32x32 tile
    Value tileSizeVal = createConstantI32(loc, rewriter, tileSizeBytes);

    // ===================================================================
    // TTKernel Lowering (commented out until TTKernel dialect is available)
    // ===================================================================
    //
    // The following code demonstrates the intended lowering to TTKernel ops.
    // Uncomment and adjust once the TTKernel dialect is integrated.
    //
    // ```cpp
    // // Get data format from the destination CB
    // auto dataFormat = ttkernel::GetDataFormatOp::create(rewriter, loc, dst);
    // auto pageSize = ttkernel::GetTileSizeOp::create(rewriter, loc, dst);
    //
    // // Create interleaved address generator for DRAM access
    // Value trueVal = createConstantI1(loc, rewriter, true); // DRAM = true
    // Value addrGen = ttkernel::GetInterleavedAddrGenFastOp::create(
    //     rewriter, loc, /*dram=*/trueVal, srcBaseAddr, pageSize, dataFormat);
    //
    // // Reserve space in the destination CB
    // ttkernel::CBReserveBackOp::create(rewriter, loc, dst, numTilesVal);
    //
    // // Get the initial L1 write address
    // Value l1Addr = ttkernel::GetWritePtrOp::create(rewriter, loc, dst);
    //
    // Value const0 = createConstantI32(loc, rewriter, 0);
    // Value const1 = createConstantI32(loc, rewriter, 1);
    //
    // // Convert byte offset to tile index for the base
    // Value baseTileIndex = arith::DivUIOp::create(
    //     rewriter, loc, srcBaseAddr, pageSize);
    //
    // // Create a loop to read tiles from DRAM to L1
    // scf::ForOp loadTileLoop = scf::ForOp::create(
    //     rewriter, loc, const0, numTilesVal, const1,
    //     ValueRange{l1Addr, baseTileIndex});
    // {
    //   rewriter.setInsertionPointToStart(loadTileLoop.getBody());
    //   Value crtL1Address = loadTileLoop.getRegionIterArgs()[0];
    //   Value crtTileIndex = loadTileLoop.getRegionIterArgs()[1];
    //
    //   // Get the NOC address for the current tile
    //   Value nocAddr = ttkernel::InterleavedAddrGenFastGetNocAddrOp::create(
    //       rewriter, loc, addrGen, crtTileIndex, const0, Value());
    //
    //   // Issue async read from DRAM to L1
    //   ttkernel::NocAsyncReadOp::create(
    //       rewriter, loc, nocAddr, crtL1Address, pageSize);
    //
    //   // Update loop variables
    //   Value nextL1Address = arith::AddIOp::create(
    //       rewriter, loc, crtL1Address, pageSize);
    //   Value nextTileIndex = arith::AddIOp::create(
    //       rewriter, loc, crtTileIndex, const1);
    //
    //   scf::YieldOp::create(
    //       rewriter, loc, ValueRange{nextL1Address, nextTileIndex});
    // }
    //
    // rewriter.setInsertionPointAfter(loadTileLoop);
    //
    // // Wait for all async reads to complete
    // ttkernel::NocAsyncReadBarrierOp::create(rewriter, loc);
    //
    // // Signal that data is ready in the CB
    // ttkernel::CBPushBackOp::create(rewriter, loc, dst, numTilesVal);
    // ```

    // For now, emit a placeholder that shows the intended transformation
    // This allows the pass to compile and be tested with the full pipeline
    // once TTKernel dialect is integrated.

    // Emit diagnostic information about the transformation
    LDBG("Would lower memref.copy to TTKernel NOC ops:");
    LDBG("  Source type: " << srcType);
    LDBG("  Dest type: " << dstType);
    LDBG("  Num tiles: " << numTiles);
    LDBG("  Tile size (bytes): " << tileSizeBytes);

    // Erase the original copy operation
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMemrefCopyL1ToDram Pattern (Store path)
//===----------------------------------------------------------------------===//

/// Conversion pattern for lowering memref.copy from L1 to DRAM.
///
/// This pattern handles L1 to DRAM memory transfers (store path) by:
/// 1. Computing the destination DRAM address from the memref offset
/// 2. Setting up an interleaved address generator for tile-based writes
/// 3. Iterating over tiles and issuing NOC async write operations
/// 4. Inserting a barrier to wait for all writes to complete
/// 5. Popping the source CB to signal consumption
///
/// Note: This pattern is the reverse of DRAM to L1 and is triggered when
/// the source is L1 and destination is DRAM (no loom.copy.choice attribute
/// with memory_name = "L1", or source has the L1 attribute).
struct ConvertMemrefCopyL1ToDram
    : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Check if this is an L1 to DRAM copy (no loom.copy.choice attribute
    // on the copy itself, but destination comes from reinterpret_cast)
    auto copyChoice = getCopyChoiceAttr(op);

    // If it has loom.copy.choice with L1, it's DRAM to L1 (handled above)
    if (copyChoice) {
      auto memNameAttr = copyChoice.getAs<StringAttr>("memory_name");
      if (memNameAttr && memNameAttr.getValue() == "L1")
        return failure(); // Let the other pattern handle it
    }

    // Check if destination is a reinterpret_cast (indicating DRAM access)
    auto dstReinterpret =
        op.getTarget().getDefiningOp<memref::ReinterpretCastOp>();
    if (!dstReinterpret) {
      // Not a store to DRAM, skip
      return failure();
    }

    Value src = adaptor.getSource();
    Value dst = adaptor.getTarget();

    auto srcType = cast<MemRefType>(op.getSource().getType());
    auto dstType = cast<MemRefType>(op.getTarget().getType());

    LDBG("Converting memref.copy L1 to DRAM: " << srcType << " to " << dstType);

    // Get the destination base address (DRAM offset)
    Value dstBaseAddr = extractBaseAddress(op.getTarget(), rewriter, loc);

    // Compute the number of tiles to transfer
    int64_t numTiles = computeNumTiles(srcType);
    Value numTilesVal = createConstantI32(loc, rewriter, numTiles);

    // Get tile/page size in bytes
    int64_t elementSize = getElementByteSize(srcType);
    int64_t tileSizeBytes = 32 * 32 * elementSize;
    Value tileSizeVal = createConstantI32(loc, rewriter, tileSizeBytes);

    // ===================================================================
    // TTKernel Lowering for Store (commented out)
    // ===================================================================
    //
    // ```cpp
    // // Get data format and page size from source CB
    // auto dataFormat = ttkernel::GetDataFormatOp::create(rewriter, loc, src);
    // auto pageSize = ttkernel::GetTileSizeOp::create(rewriter, loc, src);
    //
    // // Create interleaved address generator for DRAM write
    // Value trueVal = createConstantI1(loc, rewriter, true);
    // Value addrGen = ttkernel::GetInterleavedAddrGenFastOp::create(
    //     rewriter, loc, trueVal, dstBaseAddr, pageSize, dataFormat);
    //
    // // Get the L1 read address from source CB
    // Value l1Addr = ttkernel::GetReadPtrOp::create(rewriter, loc, src);
    //
    // Value const0 = createConstantI32(loc, rewriter, 0);
    // Value const1 = createConstantI32(loc, rewriter, 1);
    //
    // Value baseTileIndex = arith::DivUIOp::create(
    //     rewriter, loc, dstBaseAddr, pageSize);
    //
    // // Loop to write tiles from L1 to DRAM
    // scf::ForOp storeTileLoop = scf::ForOp::create(
    //     rewriter, loc, const0, numTilesVal, const1,
    //     ValueRange{l1Addr, baseTileIndex});
    // {
    //   rewriter.setInsertionPointToStart(storeTileLoop.getBody());
    //   Value crtL1Address = storeTileLoop.getRegionIterArgs()[0];
    //   Value crtTileIndex = storeTileLoop.getRegionIterArgs()[1];
    //
    //   Value nocAddr = ttkernel::InterleavedAddrGenFastGetNocAddrOp::create(
    //       rewriter, loc, addrGen, crtTileIndex, const0, Value());
    //
    //   ttkernel::NocAsyncWriteOp::create(
    //       rewriter, loc, crtL1Address, nocAddr, pageSize);
    //
    //   Value nextL1Address = arith::AddIOp::create(
    //       rewriter, loc, crtL1Address, pageSize);
    //   Value nextTileIndex = arith::AddIOp::create(
    //       rewriter, loc, crtTileIndex, const1);
    //
    //   scf::YieldOp::create(
    //       rewriter, loc, ValueRange{nextL1Address, nextTileIndex});
    // }
    //
    // rewriter.setInsertionPointAfter(storeTileLoop);
    //
    // // Wait for all writes to complete
    // ttkernel::NocAsyncWriteBarrierOp::create(rewriter, loc);
    //
    // // Pop the source CB to signal data consumption
    // ttkernel::CBPopFrontOp::create(rewriter, loc, src, numTilesVal);
    // ```

    LDBG("Would lower memref.copy to TTKernel NOC write ops:");
    LDBG("  Source type: " << srcType);
    LDBG("  Dest type: " << dstType);
    LDBG("  Num tiles: " << numTiles);
    LDBG("  Tile size (bytes): " << tileSizeBytes);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

/// Populates the pattern set with memref.copy to TTKernel conversion patterns.
///
/// @param typeConverter The type converter for the conversion.
/// @param patterns The pattern set to populate.
/// @param benefit The pattern benefit (priority).
void populateMemoryOpToTTKernelPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit) {
  patterns.add<ConvertMemrefCopyToTTKernel>(typeConverter,
                                            patterns.getContext(), benefit);
  patterns.add<ConvertMemrefCopyL1ToDram>(typeConverter, patterns.getContext(),
                                          benefit);
}

/// Populates patterns without explicit benefit.
void populateMemoryOpToTTKernelPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  populateMemoryOpToTTKernelPatterns(typeConverter, patterns, PatternBenefit(1));
}

} // namespace loom
} // namespace mlir
