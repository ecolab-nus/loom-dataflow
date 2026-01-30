/**
 * @file MemoryOpToTTKernel.cpp
 * @brief Implementation for memory operation to TT kernel conversion pass.
 * @details
 * This pass processes memory operations whose destination allocations carry
 * `{loom.alloc ...}` attributes and uses the pre-created compile-arg values
 * from the CompileArgTracker.
 */

#include "MemoryOpToTTKernel.h"
#include "FuncOpToTTKernel.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;
using namespace tt::ttcore;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/**
 * @brief Check if the operation is inside a compute kernel function.
 *
 * @details Determines kernel type by checking the ThreadTypeAttr on the parent
 *          function. Compute kernels have ThreadType::Compute, while
 * reader/writer kernels have ThreadType::Noc.
 *
 * @param op The operation to check.
 * @return true if the operation is inside a compute kernel, false otherwise.
 */
static bool isComputeKernel(Operation *op) {
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return false;

  auto threadAttr =
      parentFunc->getAttrOfType<ThreadTypeAttr>(ThreadTypeAttr::name);
  return threadAttr && threadAttr.getValue() == ThreadType::Compute;
}

//===----------------------------------------------------------------------===//
// Memory Kernel Patterns (Reader/Writer)
//===----------------------------------------------------------------------===//

/**
 * @brief Convert `memref.copy` into TTKernel NOC read ops for reader kernels.
 *
 * @details The conversion uses CompileArgTracker to get the pre-created base
 *          address value from the source memref. It then emits a TTKernel NOC
 *          read sequence to populate the destination circular buffer (CB).
 *          This pattern is used for reader kernels (Noc thread type).
 */

std::pair<Value, Value> dram_read(memref::CopyOp op,
                                   ConversionPatternRewriter &rewriter,
                                   MemrefArgData *memrefArgData,
                                   std::shared_ptr<CompileArgTracker> tracker,
                                   const TypeConverter *typeConverter) {
  Value source = op.getSource();
  auto reinterpretCastOp = source.getDefiningOp<memref::ReinterpretCastOp>();
  if (!reinterpretCastOp)
    return std::make_pair(Value(), Value());

  Location loc = op.getLoc();

  // Get the input memref (source of reinterpret_cast) - this should be a
  // function argument.
  Value inputMemref = reinterpretCastOp.getSource();

  // Get the pre-created CB from the tracker.
  Value cb = tracker->getCB(inputMemref);
  if (!cb) {
    // Fallback: try to get from remapped value (alloc conversion may have run).
    cb = rewriter.getRemappedValue(op.getTarget());
  }
  if (!cb) {
    // Still no CB - create a fallback.
    auto cbType =
        cast<CBType>(typeConverter->convertType(op.getTarget().getType()));
    Value idxValue =
        rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
    cb = rewriter.create<GetArgValOp>(loc, cbType, idxValue);
  }

  // Get the pre-created base address from the tracker.
  Value baseAddr = tracker->getBaseAddr(inputMemref);
  if (!baseAddr) {
    // If not found in tracker (e.g., not a function argument), create one here.
    // This is a fallback for memrefs that weren't processed at function entry.
    Value baseAddrIdxValue =
        rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
    baseAddr = rewriter.create<GetArgValOp>(loc, rewriter.getI32Type(),
                                            baseAddrIdxValue);
  }

  // Determine insertion point: must be after both cb and baseAddr.
  Value insertionAnchor = cb;
  Operation *cbOp = cb.getDefiningOp();
  Operation *baseAddrOp = baseAddr.getDefiningOp();
  if (cbOp && baseAddrOp && cbOp->getBlock() == baseAddrOp->getBlock()) {
    // If both are in the same block, insert after the later one.
    if (cbOp->isBeforeInBlock(baseAddrOp)) {
      insertionAnchor = baseAddr;
    }
  }

  auto opInsertionPt = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointAfterValue(insertionAnchor);

  auto pageSize = GetTileSizeOp::create(rewriter, loc, cb);

  // Get the offset from reinterpret_cast (use first offset if multiple)
  Value offset;
  {
    auto offsets = reinterpretCastOp.getOffsets();
    if (!offsets.empty()) {
      offset = offsets[0];
    } else {
      // If no dynamic offsets, check static offsets
      auto mixedOffsets = reinterpretCastOp.getMixedOffsets();
      if (!mixedOffsets.empty() && isa<Attribute>(mixedOffsets[0])) {
        // Static offset - convert to value
        auto staticOffset =
            llvm::cast<IntegerAttr>(cast<Attribute>(mixedOffsets[0]));
        offset =
            rewriter.create<arith::ConstantIndexOp>(loc, staticOffset.getInt());
      } else {
        // Fallback: error
        llvm::errs() << "No offset found for memref.reinterpret_cast\n";
        return std::make_pair(Value(), Value());
      }
    }
  }

  rewriter.restoreInsertionPoint(opInsertionPt);

  // Convert offset to i32 for use in calculations.
  // arith.divui requires operands/results to have the same type.
  // Our `offset` is typically `index` (from memref.reinterpret_cast), while
  // TTKernel tile size is `i32`. Convert offset to `i32` before dividing.
  Value offsetI32 = offset;
  if (offsetI32 && offsetI32.getType().isIndex()) {
    offsetI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                                    offsetI32);
  }

  // Determine how many tiles we need to load by converting the shape to tiles.
  auto cbType = cast<CBType>(cb.getType());
  Value numPages;
  Value totalSizeBytes;

  // Check if CB is tiled (element type is TileType) or scalar
  auto elementType = cbType.getElementType();
  if (auto tileType = llvm::dyn_cast<TileType>(elementType)) {
    // Tiled CB: use getNumTiles() directly
    const int32_t numTiles = cbType.getNumTiles();
    numPages = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(),
                                                     numTiles);
    // For tiled CB, calculate total size: numTiles * pageSize
    totalSizeBytes = arith::MulIOp::create(rewriter, loc, numPages, pageSize);
  } else {
    // Scalar CB: calculate numTiles = (numElements * elementSizeInBytes) /
    // pageSizeInBytes Get number of elements and element size
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
    totalSizeBytes = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), numElements * elementSizeBytes);
    numPages = arith::DivUIOp::create(rewriter, loc, totalSizeBytes, pageSize);
  }
  // Get the pre-created TensorAccessor from the tracker.
  Value accessorOp = tracker->getTensorAccessor(inputMemref);
  if (!accessorOp) {
    llvm::errs() << "No TensorAccessor found for input memref\n";
    return std::make_pair(Value(), Value());
  }

  // Reserve space in CB and obtain write pointer.
  CBReserveBackOp::create(rewriter, loc, cb, numPages);
  Value l1Addr = GetWritePtrOp::create(rewriter, loc, cb);
  Value multicast_l1Addr = GetWritePtrOp::create(rewriter, loc, cb);

  // Obtain the base offset value from the reinterpret_cast operation.
  // The offset is in element units (from affine.apply).
  Value baseElemOffset = offsetI32;

  // Get memref shape.
  auto resultType = cast<MemRefType>(reinterpretCastOp.getResult().getType());
  ArrayRef<int64_t> shape = resultType.getShape();

  // Get strides from the layout.
  SmallVector<int64_t> strides;
  auto layout = resultType.getLayout();
  if (auto stridedLayout = dyn_cast<StridedLayoutAttr>(layout)) {
    auto stridesRef = stridedLayout.getStrides();
    strides.append(stridesRef.begin(), stridesRef.end());
  } else {
    // Compute default row-major strides if not available.
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides.insert(strides.begin(), stride);
      stride *= shape[i];
    }
  }

  // Tile dimension (32x32 for TTKernel).
  constexpr int64_t kTileDim = 32;

  // Calculate number of tiles in each dimension: numTiles = shape / 32.
  // For 2D memref [H, W], we have (H/32) x (W/32) tiles.
  int64_t numTileRows =
      (shape.size() > 0) ? (shape[0] + kTileDim - 1) / kTileDim : 1;
  int64_t numTileCols =
      (shape.size() > 1) ? (shape[1] + kTileDim - 1) / kTileDim : 1;

  // Create constants for loop bounds and calculations.
  Value loopConst0 =
      rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
  Value loopConst1 =
      rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 1);
  Value tileDimVal = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), kTileDim);
  Value numTileRowsVal = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), numTileRows);
  Value numTileColsVal = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), numTileCols);
  Value stride0Val = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), (strides.size() > 0) ? strides[0] : 1);
  Value stride1Val = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), (strides.size() > 1) ? strides[1] : 1);

  // Create nested loops to iterate over all tiles.
  // Outer loop: iterate over tile rows (0 to numTileRows).
  scf::ForOp rowLoop =
      scf::ForOp::create(rewriter, loc, loopConst0, numTileRowsVal, loopConst1,
                         ValueRange{l1Addr});
  {
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    Value tileRow = rowLoop.getInductionVar();
    Value crtL1Addr = rowLoop.getRegionIterArgs()[0];

    // Inner loop: iterate over tile columns (0 to numTileCols).
    scf::ForOp colLoop =
        scf::ForOp::create(rewriter, loc, loopConst0, numTileColsVal,
                           loopConst1, ValueRange{crtL1Addr});
    {
      rewriter.setInsertionPointToStart(colLoop.getBody());
      Value tileCol = colLoop.getInductionVar();
      Value innerL1Addr = colLoop.getRegionIterArgs()[0];

      // Calculate element offset for this tile within the memref:
      // tileElemOffset = tileRow * tileDim * stride0 + tileCol * tileDim *
      // stride1
      Value rowOffset =
          arith::MulIOp::create(rewriter, loc, tileRow, tileDimVal);
      rowOffset = arith::MulIOp::create(rewriter, loc, rowOffset, stride0Val);

      Value colOffset =
          arith::MulIOp::create(rewriter, loc, tileCol, tileDimVal);
      colOffset = arith::MulIOp::create(rewriter, loc, colOffset, stride1Val);

      Value tileElemOffset =
          arith::AddIOp::create(rewriter, loc, rowOffset, colOffset);

      // Total element offset = baseElemOffset + tileElemOffset
      Value totalElemOffset =
          arith::AddIOp::create(rewriter, loc, baseElemOffset, tileElemOffset);

      // rowIdx = totalElemOffset / elementsPerRow
      Value rowIdx =
          arith::DivUIOp::create(rewriter, loc, totalElemOffset, stride0Val);

      // colIdx = totalElemOffset % elementsPerRow
      Value colIdx =
          arith::RemUIOp::create(rewriter, loc, totalElemOffset, stride0Val);

      // tilesPerRow = elementsPerRow / tileDim
      Value tilesPerRow =
          arith::DivUIOp::create(rewriter, loc, stride0Val, tileDimVal);

      // rowTile = rowIdx / tileDim
      Value rowTile = arith::DivUIOp::create(rewriter, loc, rowIdx, tileDimVal);

      // rowTileBase = rowTile * tilesPerRow
      Value rowTileBase =
          arith::MulIOp::create(rewriter, loc, rowTile, tilesPerRow);

      // colTile = colIdx / tileDim
      Value colTile = arith::DivUIOp::create(rewriter, loc, colIdx, tileDimVal);

      // tileId = rowTileBase + colTile
      Value tileId = arith::AddIOp::create(rewriter, loc, rowTileBase, colTile);

      // Issue an async read for this tile using the tensor accessor.
      NocAsyncReadTileOp::create(rewriter, loc, tileId, accessorOp, innerL1Addr);

      // Advance L1 address to next tile by adding tile_size.
      Value nextL1Addr =
          arith::AddIOp::create(rewriter, loc, innerL1Addr, pageSize);
      scf::YieldOp::create(rewriter, loc, ValueRange(nextL1Addr));
    }

    // Yield updated L1 address from inner loop.
    rewriter.setInsertionPointAfter(colLoop);
    scf::YieldOp::create(rewriter, loc, colLoop.getResults());
  }
  rewriter.setInsertionPointAfter(rowLoop);

  // Barrier to wait for all async reads to complete.
  NocAsyncReadBarrierOp::create(rewriter, loc);

  return std::make_pair(totalSizeBytes, multicast_l1Addr);
}

void multicast_send(ConversionPatternRewriter &rewriter, Location loc, MemrefArgData *memrefArgData, Value totalSizeBytes, Value multicast_l1Addr) {
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
  NocSemaphoreSetOp::create(rewriter, loc,
                            memrefArgData->mcast_sender_semaphore_addr_ptr, zero);

 // Create noc_id as a Value
  Value nocIdVal = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI8Type(), memrefArgData->noc_id);
   Value noc_multicast_addr = GetNocMulticastAddrOp::create(
      rewriter, loc, NocAddrType::get(rewriter.getContext()),
      memrefArgData->mcast_dest_noc_start_x,
      memrefArgData->mcast_dest_noc_start_y,
      memrefArgData->mcast_dest_noc_end_x, memrefArgData->mcast_dest_noc_end_y,
      multicast_l1Addr, nocIdVal);
  // send the data to the destinations
  // Provide explicit false attrs when supplying noc_id to avoid asm ambiguity.
  auto falseAttr = rewriter.getBoolAttr(false);
  NocAsyncWriteMulticastOp::create(
      rewriter, loc, multicast_l1Addr, noc_multicast_addr,
      totalSizeBytes, // total size of last read
      memrefArgData->mcast_dest_num,
      falseAttr, // linked (optional BoolAttr)
      falseAttr, // multicast_path_reserve (optional BoolAttr)
      nocIdVal);
  Value noc_multicast_receiver_semaphore_addr =
      GetNocMulticastAddrOp::create(
          rewriter, loc, NocAddrType::get(rewriter.getContext()),
          memrefArgData->mcast_dest_noc_start_x,
          memrefArgData->mcast_dest_noc_start_y,
          memrefArgData->mcast_dest_noc_end_x,
          memrefArgData->mcast_dest_noc_end_y,
          memrefArgData->mcast_receiver_semaphore_addr, nocIdVal);
}

void multicast_receive(ConversionPatternRewriter &rewriter, Location loc,
                       MemrefArgData *memrefArgData) {
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
  NocSemaphoreSetOp::create(
      rewriter, loc, memrefArgData->mcast_receiver_semaphore_addr_ptr, zero);
  // get noc address of sender semaphore
  Value mcast_sender_semaphore_addr =
      GetNocAddrOp::create(rewriter, loc, memrefArgData->mcast_sender_noc_x,
                           memrefArgData->mcast_sender_noc_y,
                           memrefArgData->mcast_sender_semaphore_addr);
      // add it by 1
      Value one = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 1);
      Value nocIdVal = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI8Type(), memrefArgData->noc_id);
      NocSemaphoreIncOp::create(rewriter, loc, mcast_sender_semaphore_addr, one, nocIdVal);
      // wait all data arrived
      NocSemaphoreWaitOp::create(rewriter, loc,
          memrefArgData->mcast_receiver_semaphore_addr_ptr, one);
}

struct ConvertMemoryLoadOp : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  /**
   * @brief Construct the pattern with a type converter and context.
   * @param typeConverter Type converter for the conversion pipeline.
   * @param context MLIR context.
   * @param tracker Shared tracker for compile-arg values.
   */
  ConvertMemoryLoadOp(TypeConverter &typeConverter, MLIRContext *context,
                      std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<memref::CopyOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  /**
   * @brief Match and rewrite `memref.copy` into TTKernel NOC read operations.
   *
   * @details This pattern handles `memref.copy` operations where the source
   *          is a `memref.reinterpret_cast` (indicating data flows from
   * external memory to L1/CB). It emits TTKernel NOC async read operations to
   *          load data from DRAM into the circular buffer.
   */
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the source - it should be a reinterpret_cast.
    Value source = op.getSource();
    auto reinterpretCastOp = source.getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpretCastOp)
      return failure();

    // Get the input memref (source of reinterpret_cast) - this should be a
    // function argument.
    Value inputMemref = reinterpretCastOp.getSource();

    // Get memrefArgData from tracker
    MemrefArgData *memrefArgData = tracker->getMemrefData(inputMemref);
    if (!memrefArgData) {
      llvm::errs() << "No MemrefArgData found for input memref\n";
      return failure();
    }

    // reserve the back of the cb
    Value numTilesVal = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), memrefArgData->num_tiles);
    CBReserveBackOp::create(rewriter, loc, memrefArgData->cb, numTilesVal);

    // based on the option in memref.copy to decide which function to call
    //  Check if this is a broadcast operation
    bool isBroadcast = false;
    bool isBroadcastX = false;
    bool isBroadcastY = false;
    if (auto choiceAttr =
            op->getAttrOfType<DictionaryAttr>("loom.copy.choice")) {
      if (auto kindAttr = choiceAttr.get("kind")) {
        if (auto kindStr = llvm::dyn_cast_or_null<StringAttr>(kindAttr)) {
          if (kindStr.getValue() == "broadcast") {
            isBroadcast = true;
            // Check which dimension is used for broadcast
            if (auto dimAttr = choiceAttr.get("dim")) {
              if (auto dimStr = llvm::dyn_cast_or_null<StringAttr>(dimAttr)) {
                StringRef dimValue = dimStr.getValue();
                if (dimValue == "x") {
                  isBroadcastX = true;
                } else if (dimValue == "y") {
                  isBroadcastY = true;
                }
              }
            }
          }
        }
      }
    }

    // Call dram_read and get the return values
    auto [totalSizeBytes, multicast_l1Addr] =
        dram_read(op, rewriter, memrefArgData, tracker, getTypeConverter());

    if (isBroadcast) {
/*       auto coreList = tracker->getCoreList();
      Value zero = rewriter.create<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), 0);
      Value cond;
      if (isBroadcastX) {
        // broadcast along x dimension, check if core_x == 0
        cond = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                     coreList[0], zero);
      } else {
        // broadcast along y dimension
        cond = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                     coreList[1], zero);
      } */

      //auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{}, cond, true);
      //rewriter.setInsertionPointToStart(ifOp.thenBlock());
      // code for sender
      multicast_send(rewriter, loc, memrefArgData, totalSizeBytes, multicast_l1Addr);
      //rewriter.create<scf::YieldOp>(loc);

      //rewriter.setInsertionPointToStart(ifOp.elseBlock());
      // code for receiver
      //multicast_receive(rewriter, loc, memrefArgData);
      //rewriter.create<scf::YieldOp>(loc);

      //rewriter.setInsertionPointAfter(ifOp);
    }
    // push the data to the cb
    CBPushBackOp::create(rewriter, loc, memrefArgData->cb, numTilesVal);
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Shared tracker for compile-arg values.
  std::shared_ptr<CompileArgTracker> tracker;
};

/**
 * @brief Convert `memref.copy` into TTKernel NOC write ops for writer kernels.
 *
 * @details The conversion handles store operations where data flows from L1/CB
 *          (source) to external memory (target). It uses the pre-created base
 *          address value and emits a TTKernel NOC write sequence to transfer
 *          data from the circular buffer to DRAM.
 *          This pattern is used for writer kernels (Noc thread type).
 */
struct ConvertMemoryStoreOp : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  /**
   * @brief Construct the pattern with a type converter and context.
   * @param typeConverter Type converter for the conversion pipeline.
   * @param context MLIR context.
   * @param tracker Shared tracker for compile-arg values.
   */
  ConvertMemoryStoreOp(TypeConverter &typeConverter, MLIRContext *context,
                       std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<memref::CopyOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  /**
   * @brief Match and rewrite `memref.copy` into TTKernel NOC write operations.
   *
   * @details This pattern handles `memref.copy` operations where the source
   *          is NOT a `memref.reinterpret_cast` (indicating data flows from
   *          L1/CB to external memory). It emits TTKernel NOC async write
   *          operations to transfer data from the circular buffer to DRAM.
   */
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Store: DRAM side is the target reinterpret_cast (writing from L1/CB to
    // external memory).
    Value target = op.getTarget();
    auto reinterpretCastOp = target.getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpretCastOp)
      return failure();
    Location loc = op.getLoc();

    // Get the output memref (source of reinterpret_cast) - used for base
    // address lookup.
    Value outputMemref = reinterpretCastOp.getSource();

    // For store, the CB is tracked per *original memref argument* (the
    // reinterpret_cast source), not via the conversion rewriter's SSA mapping.
    //
    // Important: `rewriter.getRemappedValue(outputMemref)` may simply return
    // `outputMemref` itself if no mapping exists, which would leave the
    // original function argument (e.g. %arg2) still used and prevent
    // `removeAllFunctionArguments()` from erasing it.
    Value cb = tracker->getCB(outputMemref);
    if (!cb) {
      llvm::errs() << "No CB found for MemoryStoreOp outputMemref: "
                   << outputMemref << "\n";
      return failure();
    }

    // Get the pre-created base address from the tracker.
    Value baseAddr = tracker->getBaseAddr(outputMemref);
    if (!baseAddr) {
      // Fallback: create base address.
      Value baseAddrIdxValue =
          rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
      baseAddr = rewriter.create<GetArgValOp>(loc, rewriter.getI32Type(),
                                              baseAddrIdxValue);
    }

    // Determine insertion point: must be after both cb and baseAddr.
    Value insertionAnchor = cb;
    Operation *cbOp = cb.getDefiningOp();
    Operation *baseAddrOp = baseAddr.getDefiningOp();
    if (cbOp && baseAddrOp && cbOp->getBlock() == baseAddrOp->getBlock()) {
      // If both are in the same block, insert after the later one.
      if (cbOp->isBeforeInBlock(baseAddrOp)) {
        insertionAnchor = baseAddr;
      }
    }

    auto opInsertionPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(insertionAnchor);
    auto pageSize = GetTileSizeOp::create(rewriter, loc, cb);

    // Get the base address and offset from the target (external memory)
    Value offset;
    if (auto reinterpretCastOp =
            target.getDefiningOp<memref::ReinterpretCastOp>()) {
      // Get the offset from reinterpret_cast (use first offset if multiple)
      auto offsets = reinterpretCastOp.getOffsets();
      if (!offsets.empty()) {
        offset = offsets[0];
      } else {
        // If no dynamic offsets, check static offsets
        auto mixedOffsets = reinterpretCastOp.getMixedOffsets();
        if (!mixedOffsets.empty() && isa<Attribute>(mixedOffsets[0])) {
          // Static offset - convert to value
          auto staticOffset =
              llvm::cast<IntegerAttr>(cast<Attribute>(mixedOffsets[0]));
          offset = rewriter.create<arith::ConstantIndexOp>(
              loc, staticOffset.getInt());
        } else {
          // Fallback: use constant 0 (index type)
          offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        }
      }
    } else {
      // No reinterpret_cast - use offset 0
      offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    }

    rewriter.restoreInsertionPoint(opInsertionPt);

    // Convert bytes offset to i32 for calculations.
    Value offsetI32 = offset;
    if (offsetI32 && offsetI32.getType().isIndex()) {
      offsetI32 = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), offsetI32);
    }

    // Determine how many tiles we need to store.
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
      // Scalar CB: calculate numTiles = (numElements * elementSizeInBytes) /
      // pageSizeInBytes
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
        // Default: assume 4 bytes if unknown
        elementSizeBytes = 4;
      }

      // Calculate total size in bytes and divide by page size
      Value totalSizeBytes = rewriter.create<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), numElements * elementSizeBytes);
      numPages =
          arith::DivUIOp::create(rewriter, loc, totalSizeBytes, pageSize);
    }

    // Get the pre-created TensorAccessor from the tracker.
    Value accessorOp = tracker->getTensorAccessor(outputMemref);
    if (!accessorOp) {
      llvm::errs() << "No TensorAccessor found for output memref\n";
      return failure();
    }

    // Wait for data in CB and get read pointer.
    CBWaitFrontOp::create(rewriter, loc, cb, numPages);
    Value l1Addr = GetReadPtrOp::create(rewriter, loc, cb);

    // Obtain the base offset value (%13) from the reinterpret_cast operation.
    // The offset is in element units (from affine.apply).
    Value baseElemOffset = offsetI32;

    // Get memref shape.
    auto resultType = cast<MemRefType>(reinterpretCastOp.getResult().getType());
    ArrayRef<int64_t> shape = resultType.getShape();

    // Get strides from the layout.
    SmallVector<int64_t> strides;
    auto layout = resultType.getLayout();
    if (auto stridedLayout = dyn_cast<StridedLayoutAttr>(layout)) {
      auto stridesRef = stridedLayout.getStrides();
      strides.append(stridesRef.begin(), stridesRef.end());
    } else {
      // Compute default row-major strides if not available.
      int64_t stride = 1;
      for (int i = shape.size() - 1; i >= 0; --i) {
        strides.insert(strides.begin(), stride);
        stride *= shape[i];
      }
    }

    // Tile dimension (32x32 for TTKernel).
    constexpr int64_t kTileDim = 32;

    // Calculate number of tiles in each dimension: numTiles = shape / 32.
    // For 2D memref [H, W], we have (H/32) x (W/32) tiles.
    int64_t numTileRows =
        (shape.size() > 0) ? (shape[0] + kTileDim - 1) / kTileDim : 1;
    int64_t numTileCols =
        (shape.size() > 1) ? (shape[1] + kTileDim - 1) / kTileDim : 1;

    // Create constants for loop bounds and calculations.
    Value loopConst0 =
        rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
    Value loopConst1 =
        rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 1);
    Value tileDimVal = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), kTileDim);
    Value numTileRowsVal = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), numTileRows);
    Value numTileColsVal = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), numTileCols);
    Value stride0Val = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), (strides.size() > 0) ? strides[0] : 1);
    Value stride1Val = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), (strides.size() > 1) ? strides[1] : 1);

    // Create nested loops to iterate over all tiles.
    // Outer loop: iterate over tile rows (0 to numTileRows).
    scf::ForOp rowLoop =
        scf::ForOp::create(rewriter, loc, loopConst0, numTileRowsVal,
                           loopConst1, ValueRange{l1Addr});
    {
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      Value tileRow = rowLoop.getInductionVar();
      Value crtL1Addr = rowLoop.getRegionIterArgs()[0];

      // Inner loop: iterate over tile columns (0 to numTileCols).
      scf::ForOp colLoop =
          scf::ForOp::create(rewriter, loc, loopConst0, numTileColsVal,
                             loopConst1, ValueRange{crtL1Addr});
      {
        rewriter.setInsertionPointToStart(colLoop.getBody());
        Value tileCol = colLoop.getInductionVar();
        Value innerL1Addr = colLoop.getRegionIterArgs()[0];

        // Calculate element offset for this tile within the memref:
        // tileElemOffset = tileRow * tileDim * stride0 + tileCol * tileDim *
        // stride1
        Value rowOffset =
            arith::MulIOp::create(rewriter, loc, tileRow, tileDimVal);
        rowOffset = arith::MulIOp::create(rewriter, loc, rowOffset, stride0Val);

        Value colOffset =
            arith::MulIOp::create(rewriter, loc, tileCol, tileDimVal);
        colOffset = arith::MulIOp::create(rewriter, loc, colOffset, stride1Val);

        Value tileElemOffset =
            arith::AddIOp::create(rewriter, loc, rowOffset, colOffset);

        // Total element offset = baseElemOffset + tileElemOffset
        Value totalElemOffset = arith::AddIOp::create(
            rewriter, loc, baseElemOffset, tileElemOffset);

        // rowIdx = totalElemOffset / elementsPerRow
        Value rowIdx =
            arith::DivUIOp::create(rewriter, loc, totalElemOffset, stride0Val);

        // colIdx = totalElemOffset % elementsPerRow
        Value colIdx =
            arith::RemUIOp::create(rewriter, loc, totalElemOffset, stride0Val);

        // tilesPerRow = elementsPerRow / tileDim
        Value tilesPerRow =
            arith::DivUIOp::create(rewriter, loc, stride0Val, tileDimVal);

        // rowTile = rowIdx / tileDim
        Value rowTile =
            arith::DivUIOp::create(rewriter, loc, rowIdx, tileDimVal);

        // rowTileBase = rowTile * tilesPerRow
        Value rowTileBase =
            arith::MulIOp::create(rewriter, loc, rowTile, tilesPerRow);

        // colTile = colIdx / tileDim
        Value colTile =
            arith::DivUIOp::create(rewriter, loc, colIdx, tileDimVal);

        // tileId = rowTileBase + colTile
        Value tileId =
            arith::AddIOp::create(rewriter, loc, rowTileBase, colTile);
        // Issue an async write for this tile using the tensor accessor.
        NocAsyncWriteTileOp::create(rewriter, loc, tileId, accessorOp,
                                    innerL1Addr);

        // Advance L1 address to next tile.
        Value nextL1Addr =
            arith::AddIOp::create(rewriter, loc, innerL1Addr, pageSize);
        scf::YieldOp::create(rewriter, loc, ValueRange(nextL1Addr));
      }

      // Yield updated L1 address from inner loop.
      rewriter.setInsertionPointAfter(colLoop);
      scf::YieldOp::create(rewriter, loc, colLoop.getResults());
    }
    rewriter.setInsertionPointAfter(rowLoop);

    NocAsyncWriteBarrierOp::create(rewriter, loc);
    CBPopFrontOp::create(rewriter, loc, cb, numPages);

    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Shared tracker for compile-arg values.
  std::shared_ptr<CompileArgTracker> tracker;
};

//===----------------------------------------------------------------------===//
// Compute Kernel Patterns
//===----------------------------------------------------------------------===//

/**
 * @brief Convert `memref.copy` to CB synchronization for compute kernels
 * (load).
 *
 * @details Compute kernels don't have direct NOC access. Instead of performing
 *          NOC reads, compute kernels wait for the reader kernel to load data
 *          into the circular buffer via cb_wait_front.
 *          This pattern handles the "load" side (source is reinterpret_cast).
 */
struct ConvertComputeLoadOp : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  /**
   * @brief Construct the pattern with a type converter and context.
   * @param typeConverter Type converter for the conversion pipeline.
   * @param context MLIR context.
   * @param tracker Shared tracker for compile-arg values.
   */
  ConvertComputeLoadOp(TypeConverter &typeConverter, MLIRContext *context,
                       std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<memref::CopyOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  /**
   * @brief Match and rewrite `memref.copy` into CB wait operations for compute.
   *
   * @details This pattern handles `memref.copy` operations in compute kernels
   *          where the source is a `memref.reinterpret_cast`. Instead of NOC
   *          operations, it emits cb_wait_front to wait for reader kernel.
   *
   * @todo Implement cb_wait_front emission when TTKernel ops are available.
   */
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Implement cb_wait_front to wait for reader kernel to load data.
    // For now, return failure() - will be implemented in follow-up.
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Shared tracker for compile-arg values.
  std::shared_ptr<CompileArgTracker> tracker;
};

/**
 * @brief Convert `memref.copy` to CB synchronization for compute kernels
 * (store).
 *
 * @details Compute kernels don't have direct NOC access. Instead of performing
 *          NOC writes, compute kernels signal completion via cb_push_back so
 *          the writer kernel can consume the data.
 *          This pattern handles the "store" side (target is reinterpret_cast).
 */
struct ConvertComputeStoreOp : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  /**
   * @brief Construct the pattern with a type converter and context.
   * @param typeConverter Type converter for the conversion pipeline.
   * @param context MLIR context.
   * @param tracker Shared tracker for compile-arg values.
   */
  ConvertComputeStoreOp(TypeConverter &typeConverter, MLIRContext *context,
                        std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<memref::CopyOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  /**
   * @brief Match and rewrite `memref.copy` into CB push operations for compute.
   *
   * @details This pattern handles `memref.copy` operations in compute kernels
   *          where the target is a `memref.reinterpret_cast`. Instead of NOC
   *          operations, it emits cb_push_back to signal data ready for writer.
   *
   * @todo Implement cb_push_back emission when TTKernel ops are available.
   */
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the target - it should be a reinterpret_cast pointing to DRAM.
    Value target = op.getTarget();
    auto reinterpretCastOp = target.getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpretCastOp)
      return failure();

    // Get the source of the reinterpret_cast - this is the original output
    // memref (function argument like %arg2).
    Value outputMemref = reinterpretCastOp.getSource();

    // Get the CB associated with this output memref from the tracker.
    // The tracker created this CB in processInputArgs for the function
    // argument.
    Value outcb = tracker->getCB(outputMemref);
    if (!outcb || !isa<CBType>(outcb.getType())) {
      llvm::errs() << "No CB found for ComputeStoreOp target: "
                   << op.getTarget() << "\n";
      return failure();
    }

    auto outcbType = cast<CBType>(outcb.getType());
    // Use getNumElements() which works for both tiled and scalar CBs
    // (getNumTiles() just calls getNumElements() but asserts element type is
    // TileType)
    // TODO: this should be a tiled CB, now is using elements
    int32_t numTiles = static_cast<int32_t>(outcbType.getNumElements()) / 1024;
    Value outcbNumInputTilesValue =
        rewriter.create<arith::ConstantIntOp>(loc, numTiles, 32);

    CBReserveBackOp::create(rewriter, loc, outcb, outcbNumInputTilesValue);
    // commit tile regs
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);
    // pack tile using scf.for loop
    Value lowerBound = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value step = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    unsigned dstIndexOffset = 0;

    scf::ForOp packLoop = scf::ForOp::create(rewriter, loc, lowerBound,
                                             outcbNumInputTilesValue, step);
    {
      rewriter.setInsertionPointToStart(packLoop.getBody());
      Value i = packLoop.getInductionVar();

      // dstIdx = i + dstIndexOffset
      Value dstIdx = i;
      if (dstIndexOffset != 0) {
        Value offsetVal = rewriter.create<arith::ConstantIntOp>(
            loc, static_cast<int32_t>(dstIndexOffset), 32);
        dstIdx = arith::AddIOp::create(rewriter, loc, i, offsetVal);
      }
      // outIdx = i
      PackTileOp::create(rewriter, loc, dstIdx, outcb, i);
    }
    rewriter.setInsertionPointAfter(packLoop);
    // release tile regs
    TileRegsReleaseOp::create(rewriter, loc);
    CBPushBackOp::create(rewriter, loc, outcb, outcbNumInputTilesValue);
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Shared tracker for compile-arg values.
  std::shared_ptr<CompileArgTracker> tracker;
};

//===----------------------------------------------------------------------===//
// Dispatcher Patterns
//===----------------------------------------------------------------------===//

/**
 * @brief Dispatcher pattern for load operations.
 *
 * @details Routes memref.copy (load) operations to the appropriate pattern
 *          based on kernel type:
 *          - Compute kernels: delegates to ConvertComputeLoadOp
 *          - Memory kernels (reader): delegates to ConvertMemoryLoadOp
 */
struct ConvertLoadOp : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  /**
   * @brief Construct the pattern with a type converter and context.
   * @param typeConverter Type converter for the conversion pipeline.
   * @param context MLIR context.
   * @param tracker Shared tracker for compile-arg values.
   */
  ConvertLoadOp(TypeConverter &typeConverter, MLIRContext *context,
                std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<memref::CopyOp>(typeConverter, context),
        computePattern(typeConverter, context, tracker),
        memoryPattern(typeConverter, context, tracker) {}

  /**
   * @brief Match and dispatch to appropriate load pattern based on kernel type.
   */
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Must be a load operation (source is reinterpret_cast)
    Value source = op.getSource();
    if (!source.getDefiningOp<memref::ReinterpretCastOp>())
      return failure();

    if (isComputeKernel(op)) {
      return computePattern.matchAndRewrite(op, adaptor, rewriter);
    }
    return memoryPattern.matchAndRewrite(op, adaptor, rewriter);
  }

private:
  /// Pattern for compute kernel loads.
  ConvertComputeLoadOp computePattern;
  /// Pattern for memory kernel loads.
  ConvertMemoryLoadOp memoryPattern;
};

/**
 * @brief Dispatcher pattern for store operations.
 *
 * @details Routes memref.copy (store) operations to the appropriate pattern
 *          based on kernel type:
 *          - Compute kernels: delegates to ConvertComputeStoreOp
 *          - Memory kernels (writer): delegates to ConvertMemoryStoreOp
 */
struct ConvertStoreOp : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  /**
   * @brief Construct the pattern with a type converter and context.
   * @param typeConverter Type converter for the conversion pipeline.
   * @param context MLIR context.
   * @param tracker Shared tracker for compile-arg values.
   */
  ConvertStoreOp(TypeConverter &typeConverter, MLIRContext *context,
                 std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<memref::CopyOp>(typeConverter, context),
        computePattern(typeConverter, context, tracker),
        memoryPattern(typeConverter, context, tracker) {}

  /**
   * @brief Match and dispatch to appropriate store pattern based on kernel
   * type.
   */
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Must be a store operation (target is reinterpret_cast)
    Value target = op.getTarget();
    if (!target.getDefiningOp<memref::ReinterpretCastOp>())
      return failure();

    if (isComputeKernel(op)) {
      return computePattern.matchAndRewrite(op, adaptor, rewriter);
    }
    return memoryPattern.matchAndRewrite(op, adaptor, rewriter);
  }

private:
  /// Pattern for compute kernel stores.
  ConvertComputeStoreOp computePattern;
  /// Pattern for memory kernel stores.
  ConvertMemoryStoreOp memoryPattern;
};

//===----------------------------------------------------------------------===//
// Other Patterns
//===----------------------------------------------------------------------===//

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
   * @param tracker Shared tracker for compile-arg values.
   */
  ConvertAllocOp(TypeConverter &typeConverter, MLIRContext *context,
                 std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<memref::AllocOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  /**
   * @brief Find the input memref that flows into this alloc via memref.copy.
   *
   * @details Looks at the users of the alloc to find a memref.copy where
   *          this alloc is the target. Then traces back through
   * reinterpret_cast to find the original input memref (typically a function
   * argument).
   *
   * @param alloc The alloc op to analyze.
   * @return The source memref if found, or nullptr.
   */
  Value findInputMemref(memref::AllocOp alloc) const {
    for (Operation *user : alloc.getResult().getUsers()) {
      if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
        // Check if this alloc is the target (destination) of the copy.
        if (copyOp.getTarget() == alloc.getResult()) {
          // Get the source - it should be a reinterpret_cast.
          Value source = copyOp.getSource();
          if (auto reinterpretCastOp =
                  source.getDefiningOp<memref::ReinterpretCastOp>()) {
            // Return the source of the reinterpret_cast (the original memref).
            return reinterpretCastOp.getSource();
          }
        }
      }
    }
    return nullptr;
  }

  /**
   * @brief Match and rewrite `memref.alloc` to TTKernel CB.
   *
   * @details This pattern converts allocations to TTKernel circular buffers.
   *          - For allocs with `loom.alloc` that receive data from an input
   * memref, use the pre-created CB from the tracker.
   *          - For other allocs (like output accumulators), create a new CB.
   */
  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Try to find the input memref that flows into this alloc.
    // If found, use the pre-created CB from the tracker.
    Value inputMemref = findInputMemref(op);
    Value cb;
    if (inputMemref) {
      cb = tracker->getCB(inputMemref);
    }

    if (!cb) {
      // No pre-created CB found. Create a new one.
      // For allocs with loom.alloc: this is a fallback
      // For allocs without loom.alloc (e.g., output accumulators): create new
      // CB
      Value idxValue = rewriter.create<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(),
          0); // TODO: use tracker to get unique index
      auto cbType =
          cast<CBType>(typeConverter->convertType(op.getResult().getType()));
      cb = rewriter.create<GetArgValOp>(loc, cbType, idxValue);
    }

    // Replace all uses of the alloc result with the CB value
    rewriter.replaceOp(op, cb);
    return success();
  }

private:
  /// Shared tracker for compile-arg values.
  std::shared_ptr<CompileArgTracker> tracker;
};

void loom::populateMemoryOpConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context, std::shared_ptr<CompileArgTracker> tracker) {
  patterns.add<ConvertAllocOp>(typeConverter, context, tracker);
  patterns.add<ConvertLoadOp>(typeConverter, context, tracker);
  patterns.add<ConvertStoreOp>(typeConverter, context, tracker);
  patterns.add<ConvertReuseReinterpretCastOp>(typeConverter, context);
}
