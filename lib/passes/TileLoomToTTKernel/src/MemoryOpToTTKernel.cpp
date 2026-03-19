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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <optional>

// Loom dialect headers for ::::loom::AllocOp, ::::loom::CopyOp
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;
using namespace tt::ttcore;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kReductionScaleCbAttrName =
    "loom.reduction_scale_cb";

/**
 * @brief Compute ceiling division of a dimension by the tile size (32).
 *
 * @param value Dimension size to be divided.
 * @return Number of tiles needed to cover the dimension, or nullopt for
 *         non-positive values.
 */
static std::optional<int64_t> ceilDiv32(int64_t value) {
  if (value <= 0)
    return std::nullopt;
  return (value + 31) / 32;
}

/**
 * @brief Compute the total number of 32x32 tiles needed for a shaped type.
 *
 * @details For each static dimension $d_i$ of the shaped type, this computes
 *          $\lceil d_i / 32 \rceil$ and returns the product across all
 *          dimensions. Dynamic or non-shaped types yield std::nullopt.
 *
 * @param type The shaped type (e.g., tensor/memref) describing the logical data.
 * @return Total number of tiles required, or nullopt if the shape is not
 *         statically known.
 */
static std::optional<int64_t> getNumTilesFromShapedType(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return std::nullopt;

  int64_t tiles = 1;
  for (int64_t dim : shaped.getShape()) {
    auto dimTiles = ceilDiv32(dim);
    if (!dimTiles)
      return std::nullopt;
    tiles *= *dimTiles;
  }
  return tiles;
}

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

static bool isWriterKernel(Operation *op) {
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return false;
  StringRef name = parentFunc.getName();
  return name.ends_with("__writer");
}

static int64_t getPackedWordForScalarOne(Type elementType) {
  if (auto tileType = dyn_cast_or_null<TileType>(elementType)) {
    switch (tileType.getDataType()) {
    case DataType::BFloat16:
      return 0x3F803F80;
    case DataType::Float16:
      return 0x3C003C00;
    case DataType::Float32:
      return 0x3F800000;
    case DataType::UInt16:
      return 0x00010001;
    case DataType::UInt8:
    case DataType::Bool:
      return 0x01010101;
    case DataType::UInt32:
    case DataType::Int32:
      return 0x00000001;
    default:
      return 0x00000001;
    }
  }

  if (elementType.isBF16())
    return 0x3F803F80;
  // This flow currently materializes scalar f16 CB payloads in Float16_b
  // runtime buffers, so scalar f16 uses packed bf16 encoding for 1.0.
  if (elementType.isF16())
    return 0x3F803F80;
  if (elementType.isF32())
    return 0x3F800000;

  if (auto intType = dyn_cast_or_null<IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 1:
    case 8:
      return 0x01010101;
    case 16:
      return 0x00010001;
    case 32:
      return 0x00000001;
    default:
      return 0x00000001;
    }
  }

  return 0x00000001;
}

static void emitReductionScaleCbInit(ConversionPatternRewriter &rewriter,
                                     Location loc, Value cb) {
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
  Value four = rewriter.create<arith::ConstantIntOp>(loc, 4, 32);
  Type elementType;
  if (auto cbType = dyn_cast_or_null<CBType>(cb.getType()))
    elementType = cbType.getElementType();
  Value encodedOneWord = rewriter.create<arith::ConstantIntOp>(
      loc, getPackedWordForScalarOne(elementType), 32);

  CBReserveBackOp::create(rewriter, loc, cb, one);
  Value writePtr = GetWritePtrOp::create(rewriter, loc, cb);
  Value l1Ptr = CastToL1PtrOp::create(rewriter, loc, writePtr);
  Value tileSizeBytes = GetTileSizeOp::create(rewriter, loc, cb);
  Value wordCount = rewriter.create<arith::DivSIOp>(loc, tileSizeBytes, four);

  scf::ForOp fillLoop = scf::ForOp::create(rewriter, loc, zero, wordCount, one);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(fillLoop.getBody());
    Value wordOffset = fillLoop.getInductionVar();
    StoreToL1Op::create(rewriter, loc, encodedOneWord, l1Ptr, wordOffset);
  }

  CBPushBackOp::create(rewriter, loc, cb, one);
}

/**
 * @brief Strip intermediate memref.cast ops to recover the original memref.
 *
 * @param value Value that may be defined by one or more `memref.cast` ops.
 * @return The first non-`memref.cast` source value.
 */
static Value stripMemrefCasts(Value value) {
  Value current = value;
  while (auto cast = current.getDefiningOp<memref::CastOp>())
    current = cast.getSource();
  return current;
}

/**
 * @brief Check whether a loom.copy is an L1->DRAM store.
 */
static bool isL1ToDramStoreCopy(::loom::CopyOp copyOp) {
  return copyOp.getDestination().getDefiningOp<memref::ReinterpretCastOp>() !=
         nullptr;
}

/**
 * @brief Find an adjacent semaphore_give paired with an L1->DRAM copy source.
 *
 * @details Matches the common pattern:
 *          `loom.copy %src, %dst_rc ...`
 *          `loom.semaphore_give %src`
 *          allowing optional intervening `memref.cast` ops.
 */
static ::loom::SemaphoreGiveOp
findAdjacentSemaphoreGiveAfterStore(::loom::CopyOp copyOp) {
  if (!isL1ToDramStoreCopy(copyOp))
    return nullptr;

  Value copySource = stripMemrefCasts(copyOp.getSource());
  Operation *next = copyOp->getNextNode();
  while (next && isa<memref::CastOp>(next))
    next = next->getNextNode();

  auto giveOp = dyn_cast_or_null<::loom::SemaphoreGiveOp>(next);
  if (!giveOp)
    return nullptr;
  if (stripMemrefCasts(giveOp.getSource()) != copySource)
    return nullptr;
  return giveOp;
}

/**
 * @brief Check if semaphore_give is paired with a preceding L1->DRAM copy.
 */
static bool isSemaphoreGiveForAdjacentL1ToDramStore(
    ::loom::SemaphoreGiveOp giveOp) {
  Operation *prev = giveOp->getPrevNode();
  while (prev && isa<memref::CastOp>(prev))
    prev = prev->getPrevNode();

  auto copyOp = dyn_cast_or_null<::loom::CopyOp>(prev);
  if (!copyOp || !isL1ToDramStoreCopy(copyOp))
    return false;

  return stripMemrefCasts(copyOp.getSource()) ==
         stripMemrefCasts(giveOp.getSource());
}

/**
 * @brief Emit TTKernel NOC async read operations for a DRAM-to-L1 transfer.
 *
 * @details Given the source value (which must come from a
 *          memref.reinterpret_cast), this function uses the CompileArgTracker
 *          to obtain pre-created CB/base-address/tensor-accessor metadata and
 *          emits a tiled NOC read loop that populates the destination CB.
 *
 * @param source The source Value (result of a memref.reinterpret_cast).
 * @param loc    Location for newly created operations.
 * @param rewriter          The conversion pattern rewriter.
 * @param memrefArgData     Pre-computed multicast/broadcast metadata.
 * @param tracker           Shared compile-arg tracker.
 * @param typeConverter     The type converter in use.
 * @return A pair of (totalSizeBytes, multicast_l1Addr), or (null, null) on
 *         error.
 */
std::pair<Value, Value> dram_read(Value source, Location loc,
                                   ConversionPatternRewriter &rewriter,
                                   MemrefArgData *memrefArgData,
                                   std::shared_ptr<CompileArgTracker> tracker,
                                   const TypeConverter *typeConverter) {
  auto reinterpretCastOp = source.getDefiningOp<memref::ReinterpretCastOp>();
  if (!reinterpretCastOp)
    return std::make_pair(Value(), Value());

  // Get the input memref (source of reinterpret_cast) - this should be a
  // function argument.
  Value inputMemref = reinterpretCastOp.getSource();

  // Get the pre-created CB from the tracker.
  Value cb = tracker->getCB(inputMemref);
  if (!cb) {
    llvm::errs() << "Error: CB not found for memref " << inputMemref << "\n";
    return std::make_pair(Value(), Value());
  }


  // Get the pre-created base address from the tracker.
  Value baseAddr = tracker->getBaseAddr(inputMemref);
  if (!baseAddr) {
    llvm::errs() << "Error: base address not found for memref " << inputMemref << "\n";
    return std::make_pair(Value(), Value());
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

  // Use num_tiles from memrefArgData (pre-computed by caller for domination correctness)
  // If not set yet (non-broadcast case), compute it now.
  Value numPages = memrefArgData->num_tiles;
  Value totalSizeBytes = arith::MulIOp::create(rewriter, loc, numPages, pageSize);

  // Get the pre-created TensorAccessor from the tracker.
  Value accessorOp = tracker->getTensorAccessor(inputMemref);
  if (!accessorOp) {
    llvm::errs() << "No TensorAccessor found for input memref\n";
    return std::make_pair(Value(), Value());
  }

  // Reserve space in CB and obtain write pointer.
  Value l1Addr = GetWritePtrOp::create(rewriter, loc, cb);
  Value multicast_l1Addr = GetWritePtrOp::create(rewriter, loc, cb);

  // Obtain the base offset value from the reinterpret_cast operation.
  // The offset is in element units.
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
      (shape.size() > 0) ? (shape[shape.size() - 2] + kTileDim - 1) / kTileDim : 1;
  int64_t numTileCols =
      (shape.size() > 1) ? (shape[shape.size() - 1] + kTileDim - 1) / kTileDim : 1;

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
    loc, rewriter.getI32Type(), (strides.size() > 0) ? strides[strides.size() - 2] : 1);
  Value stride1Val = rewriter.create<arith::ConstantIntOp>(
    loc, rewriter.getI32Type(), (strides.size() > 1) ? strides[strides.size() - 1] : 1);

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
          arith::MulIOp::create(rewriter, loc, tileRow, tileDimVal,
                                arith::IntegerOverflowFlags::nsw);
      rowOffset = arith::MulIOp::create(rewriter, loc, rowOffset, stride0Val,
                                        arith::IntegerOverflowFlags::nsw);

      Value colOffset =
          arith::MulIOp::create(rewriter, loc, tileCol, tileDimVal,
                                arith::IntegerOverflowFlags::nsw);
      colOffset = arith::MulIOp::create(rewriter, loc, colOffset, stride1Val,
                                        arith::IntegerOverflowFlags::nsw);

      Value tileElemOffset =
          arith::AddIOp::create(rewriter, loc, rowOffset, colOffset,
                                arith::IntegerOverflowFlags::nsw);

      // Total element offset = baseElemOffset + tileElemOffset
      Value totalElemOffset =
          arith::AddIOp::create(rewriter, loc, baseElemOffset, tileElemOffset,
                                arith::IntegerOverflowFlags::nsw);

      // rowIdx = totalElemOffset / elementsPerRow
      Value rowIdx =
          arith::DivSIOp::create(rewriter, loc, totalElemOffset, stride0Val);

      // colIdx = totalElemOffset % elementsPerRow
      Value colIdx =
          arith::RemSIOp::create(rewriter, loc, totalElemOffset, stride0Val);

      // tilesPerRow = elementsPerRow / tileDim
      Value tilesPerRow =
          arith::DivSIOp::create(rewriter, loc, stride0Val, tileDimVal);

      // rowTile = rowIdx / tileDim
      Value rowTile = arith::DivSIOp::create(rewriter, loc, rowIdx, tileDimVal);

      // rowTileBase = rowTile * tilesPerRow
      Value rowTileBase =
          arith::MulIOp::create(rewriter, loc, rowTile, tilesPerRow,
                                arith::IntegerOverflowFlags::nsw);

      // colTile = colIdx / tileDim
      Value colTile = arith::DivSIOp::create(rewriter, loc, colIdx, tileDimVal);

      // tileId = rowTileBase + colTile
      Value tileId = arith::AddIOp::create(rewriter, loc, rowTileBase, colTile,
                                           arith::IntegerOverflowFlags::nsw);

      // Issue an async read for this tile using the tensor accessor.
      NocAsyncReadTileOp::create(rewriter, loc, tileId, accessorOp, innerL1Addr);

      // Advance L1 address to next tile by adding tile_size.
      Value nextL1Addr =
          arith::AddIOp::create(rewriter, loc, innerL1Addr, pageSize,
                                arith::IntegerOverflowFlags::nsw);
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

//it seems this only works for host kernels, dataflow kernel doesn't include worker_logical_row_to_virtual_row
LogicalResult generate_multicast_address(
    bool isBroadcastX, ConversionPatternRewriter &rewriter, Location loc,
    MemrefArgData *memrefArgData, std::shared_ptr<CompileArgTracker> tracker,
    Operation *parentFunc, Value &logicalCoreX, Value &logicalCoreY) {
  if (!memrefArgData)
    return failure();
  Location initLoc =
      memrefArgData->initLoc ? Location(memrefArgData->initLoc) : loc;

  logicalCoreX = tracker->getCoreCoordForDim(parentFunc, "x");
  logicalCoreY = tracker->getCoreCoordForDim(parentFunc, "y");
  if (!logicalCoreX || !logicalCoreY) {
    llvm::errs() << "missing mapped spatial core coordinates for broadcast; "
                    "expected loom.mapped_to_dims to define both @x and @y\n";
    return failure();
  }

  {
    // Hoist multicast-address helpers to where core compile-args are
    // available (outside loop bodies), rather than emitting at loom.copy.
    OpBuilder::InsertionGuard guard(rewriter);
    Operation *insertAfter = nullptr;

    auto considerInsertionAnchor = [&](Value value) {
      if (Operation *defOp = value.getDefiningOp()) {
        if (!insertAfter ||
            (insertAfter->getBlock() == defOp->getBlock() &&
             insertAfter->isBeforeInBlock(defOp)))
          insertAfter = defOp;
      }
    };
    considerInsertionAnchor(logicalCoreX);
    considerInsertionAnchor(logicalCoreY);
    if (insertAfter)
      rewriter.setInsertionPointAfter(insertAfter);

    auto toI32 = [&](Value v) -> Value {
      if (v.getType().isIndex())
        return rewriter.create<arith::IndexCastOp>(initLoc,
                                                   rewriter.getI32Type(), v);
      return v;
    };

    logicalCoreX = toI32(logicalCoreX);
    logicalCoreY = toI32(logicalCoreY);

    Value zero = rewriter.create<arith::ConstantIntOp>(initLoc,
                                                       rewriter.getI32Type(), 0);
    Value one = rewriter.create<arith::ConstantIntOp>(initLoc,
                                                      rewriter.getI32Type(), 1);
    // TODO: replace this with a runtime/core-grid-derived max coordinate.
    Value seven = rewriter.create<arith::ConstantIntOp>(
        initLoc, rewriter.getI32Type(), 7);

    Value currentPhysicalCoreX =
        rewriter.create<ConvertLogicalXToTranslatedOp>(initLoc,
                                                       rewriter.getI32Type(),
                                                       logicalCoreX);
    Value currentPhysicalCoreY =
        rewriter.create<ConvertLogicalYToTranslatedOp>(initLoc,
                                                       rewriter.getI32Type(),
                                                       logicalCoreY);

    Value senderCoreX;
    Value senderCoreY;
    Value startCoreX;
    Value startCoreY;
    Value endCoreX;
    Value endCoreY;
    if (isBroadcastX) {
      // Broadcast to all cores in the same column (same x).
      senderCoreX = currentPhysicalCoreX;
      senderCoreY = rewriter.create<ConvertLogicalYToTranslatedOp>(
          initLoc, rewriter.getI32Type(), zero);
      startCoreX = currentPhysicalCoreX;
      startCoreY = rewriter.create<ConvertLogicalYToTranslatedOp>(
          initLoc, rewriter.getI32Type(), one);
      endCoreY = rewriter.create<ConvertLogicalYToTranslatedOp>(
          initLoc, rewriter.getI32Type(), seven);
      endCoreX = currentPhysicalCoreX;
    } else {
      // Broadcast to all cores in the same row (same y).
      senderCoreX = rewriter.create<ConvertLogicalXToTranslatedOp>(
          initLoc, rewriter.getI32Type(), zero);
      senderCoreY = currentPhysicalCoreY;
      startCoreX = rewriter.create<ConvertLogicalXToTranslatedOp>(
          initLoc, rewriter.getI32Type(), one);
      startCoreY = currentPhysicalCoreY;
      endCoreX = rewriter.create<ConvertLogicalXToTranslatedOp>(
          initLoc, rewriter.getI32Type(), seven);
      endCoreY = currentPhysicalCoreY;
    }

    memrefArgData->mcast_sender_noc_x = senderCoreX;
    memrefArgData->mcast_sender_noc_y = senderCoreY;
    memrefArgData->mcast_dest_noc_start_x = startCoreX;
    memrefArgData->mcast_dest_noc_start_y = startCoreY;
    memrefArgData->mcast_dest_noc_end_x = endCoreX;
    memrefArgData->mcast_dest_noc_end_y = endCoreY;
  }
  return success();
}

bool multicast_send(ConversionPatternRewriter &rewriter, Location loc, MemrefArgData *memrefArgData, Value totalSizeBytes, Value multicast_l1Addr) {
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
  auto falseAttr = rewriter.getBoolAttr(false);
  // Create noc_id as a Value
  Value nocIdVal = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI8Type(), memrefArgData->noc_id);
  Value noc_multicast_addr = GetNocMulticastAddrOp::create(
      rewriter, loc, NocAddrType::get(rewriter.getContext()),
      memrefArgData->mcast_dest_noc_end_x,
      memrefArgData->mcast_dest_noc_end_y,
      memrefArgData->mcast_dest_noc_start_x, 
      memrefArgData->mcast_dest_noc_start_y,
      multicast_l1Addr, nocIdVal);

  //init multicast semaphore
  // Store 1 to the semaphore pointer: *(mcast_receiver_semaphore_addr_ptr) = 1;
  Value oneValue = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 1);
  //TODO, should only work for valid L1 addresses, need to consider how to mantain the parameters of each memref input
  StoreToL1Op::create(rewriter, memrefArgData->initLoc, oneValue, memrefArgData->mcast_receiver_semaphore_addr_ptr, zero);
  // FIRST: wait for all destinations to be ready (receivers increment this semaphore)
  NocSemaphoreWaitOp::create(rewriter, loc, memrefArgData->mcast_sender_semaphore_addr_ptr, memrefArgData->mcast_dest_num);
  // THEN: reset the semaphore to 0 for the next iteration
  NocSemaphoreSetOp::create(rewriter, loc, memrefArgData->mcast_sender_semaphore_addr_ptr, zero);
  // send the data to the destinations
  // Provide explicit false attrs when supplying noc_id to avoid asm ambiguity.
  NocAsyncWriteMulticastOp::create(
      rewriter, loc, multicast_l1Addr, noc_multicast_addr,
      totalSizeBytes, // total size of last read
      memrefArgData->mcast_dest_num,
      falseAttr, // linked (optional BoolAttr)
      falseAttr, // multicast_path_reserve (optional BoolAttr)
      nocIdVal);

  NocSemaphoreSetMulticastOp::create(rewriter, loc, 
    memrefArgData->mcast_receiver_semaphore_addr, 
    memrefArgData->mcast_receiver_semaphore_noc_addr,
    memrefArgData->mcast_dest_num,
    falseAttr,  // linked (optional BoolAttr)
    falseAttr); // multicast_path_reserve (optional BoolAttr)
  return true;
}

bool multicast_receive(ConversionPatternRewriter &rewriter, Location loc,
                       MemrefArgData *memrefArgData) {
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
  NocSemaphoreSetOp::create(
      rewriter, loc, memrefArgData->mcast_receiver_semaphore_addr_ptr, zero);
      // add it by 1
      Value one = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 1);
      Value nocIdVal = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI8Type(), memrefArgData->noc_id);
      NocSemaphoreIncOp::create(rewriter, loc, memrefArgData->mcast_sender_semaphore_noc_addr, one, nocIdVal);
      // wait all data arrived
      NocSemaphoreWaitOp::create(rewriter, loc,
          memrefArgData->mcast_receiver_semaphore_addr_ptr, one);
  return true;
}

//===----------------------------------------------------------------------===//
// Other Patterns
//===----------------------------------------------------------------------===//

/**
 * @brief Erase memref.reinterpret_cast ops during conversion.
 *
 * @details After the copy conversion patterns have consumed the
 *          reinterpret_cast results, these ops become dead and must be removed
 *          because they are marked as illegal in the conversion target.
 *          This pattern handles both casts carrying the `loom.reuse` attribute
 *          and plain casts that were used by loom.copy / memref.copy.
 */
struct ConvertReinterpretCastOp
    : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Loom Dialect Patterns (loom.alloc, loom.semaphore, loom.copy)
//===----------------------------------------------------------------------===//

/**
 * @brief Convert `loom.semaphore` to a fresh runtime-arg CB.
 *
 * @details Each semaphore materializes an independent CB handle. This pattern
 *          allocates a new compile/runtime arg (GetArgValOp-backed) per
 *          `loom.semaphore`, even when multiple semaphores share the same
 *          physical source buffer. it first check whether the semaphore is the source of a DRAM->L1 copy, if so, return the source memref arg.
 *          otherwise, return the destination memref arg. else, create a new compile-arg CB.
 */
struct ConvertLoomSemaphoreTakeOp
    : public OpConversionPattern<::loom::SemaphoreTakeOp> {
  using OpConversionPattern<::loom::SemaphoreTakeOp>::OpConversionPattern;

  /**
   * @brief Construct the pattern with a type converter and context.
   * @param typeConverter Type converter for the conversion pipeline.
   * @param context MLIR context.
   * @param tracker Shared tracker for compile-arg values.
   */
  ConvertLoomSemaphoreTakeOp(TypeConverter &typeConverter, MLIRContext *context,
                         std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<::loom::SemaphoreTakeOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  /// If semaphore is destination of DRAM->L1 copy, return the source memref arg.
  Value findInputMemref(::loom::SemaphoreTakeOp semaphore) const {
    for (Operation *user : semaphore.getResult().getUsers()) {
      auto loomCopyOp = dyn_cast<::loom::CopyOp>(user);
      if (!loomCopyOp || loomCopyOp.getDestination() != semaphore.getResult())
        continue;

      Value source = stripMemrefCasts(loomCopyOp.getSource());
      if (auto reinterpretCastOp =
              source.getDefiningOp<memref::ReinterpretCastOp>()) {
        return stripMemrefCasts(reinterpretCastOp.getSource());
      }
    }
    return {};
  }

  /// If semaphore is source of L1->DRAM copy, return the destination memref arg.
  Value findOutputMemref(::loom::SemaphoreTakeOp semaphore) const {
    for (Operation *user : semaphore.getResult().getUsers()) {
      auto loomCopyOp = dyn_cast<::loom::CopyOp>(user);
      if (!loomCopyOp || loomCopyOp.getSource() != semaphore.getResult())
        continue;

      Value destination = stripMemrefCasts(loomCopyOp.getDestination());
      if (auto reinterpretCastOp =
              destination.getDefiningOp<memref::ReinterpretCastOp>()) {
        return stripMemrefCasts(reinterpretCastOp.getSource());
      }
    }
    return {};
  }

  LogicalResult
  matchAndRewrite(::loom::SemaphoreTakeOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    bool isReductionScaleCb = op->hasAttr(kReductionScaleCbAttrName);
    bool isDataMovementScaleInitTarget =
        isReductionScaleCb && isWriterKernel(op);

    // semaphore_take participates in CB handle materialization only for
    // compute kernels. In reader/writer/host kernels, keep the underlying
    // memref flow so loom.copy rewrites can consume it and erase the op.
    if (!isComputeKernel(op) && !isDataMovementScaleInitTarget) {
      rewriter.replaceOp(op, op.getSource());
      return success();
    }

    Location loc = op.getLoc();
    auto expectedCBType = dyn_cast_or_null<CBType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto memrefType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!memrefType)
      return rewriter.notifyMatchFailure(
          op, "loom.semaphore result must be a ranked memref type");
    CBType defaultCBType =
        expectedCBType ? expectedCBType : CBType::get(memrefType);
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc)
      return rewriter.notifyMatchFailure(op,
                                         "loom.semaphore must be inside func.func");
    if (!tracker)
      return rewriter.notifyMatchFailure(op, "compile-arg tracker is null");

    // Prefer reusing already-created memref-argument CB indexes.
    Value cb;
    if (Value inputMemref = findInputMemref(op))
      cb = tracker->getCB(inputMemref);

    if (!cb) {
      if (Value outputMemref = findOutputMemref(op))
        cb = tracker->getCB(outputMemref);
    }

    // Fallback for internal-only semaphore buffers not tied to memref args.
    if (!cb) {
      cb = tracker->createTypedCompileArg(loc, rewriter, parentFunc, defaultCBType);
    }
    if (!cb)
      return rewriter.notifyMatchFailure(
          op, "failed to create compile-arg CB for loom.semaphore");

    if (expectedCBType && cb.getType() != expectedCBType) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "CB type mismatch for loom.semaphore replacement, expected "
             << expectedCBType << " but got " << cb.getType();
      });
    }

    if (isDataMovementScaleInitTarget)
      emitReductionScaleCbInit(rewriter, loc, cb);

    rewriter.replaceOp(op, cb);
    return success();
  }

private:
  /// Shared tracker for compile-arg values.
  std::shared_ptr<CompileArgTracker> tracker;
};

/**
 * @brief Lower `loom.semaphore_give` to `ttkernel.cb_pop_front`.
 *
 * @details The semaphore operand is already type-converted to a TTKernel CB.
 *          Releasing the semaphore corresponds to consuming the front pages of
 *          that CB.
 */
struct ConvertLoomSemaphoreGiveOp
    : public OpConversionPattern<::loom::SemaphoreGiveOp> {
  using OpConversionPattern<::loom::SemaphoreGiveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::loom::SemaphoreGiveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // L1->DRAM stores are drained by writer kernels. The adjacent give is
    // only a liveness marker and must not lower to cb_pop_front in compute.
    if (isSemaphoreGiveForAdjacentL1ToDramStore(op)) {
      rewriter.eraseOp(op);
      return success();
    }

    // semaphore_give maps to CB release only for compute kernels.
    // In reader/writer/host kernels, it is a liveness marker only.
    if (!isComputeKernel(op)) {
      rewriter.eraseOp(op);
      return success();
    }

    Location loc = op.getLoc();
    Value cb = adaptor.getSource();
    auto cbType = dyn_cast_or_null<CBType>(cb.getType());
    if (!cbType) {
      return rewriter.notifyMatchFailure(
          op, "expected semaphore_give source to be converted to CB type");
    }

    Value numPages;
    auto tilesOpt = getNumTilesFromShapedType(op.getSource().getType());
    int64_t numTiles =
        tilesOpt ? *tilesOpt : static_cast<int64_t>(cbType.getNumTiles());

    numPages = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), numTiles);


    CBPopFrontOp::create(rewriter, loc, cb, numPages);
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Erase dead `loom.alloc` once downstream users are rewritten.
 *
 * @details `loom.alloc` must stay available while semaphore/copy patterns
 *          discover CB mappings. After those rewrites run, dead allocs are
 *          simply removed instead of materializing placeholder memref.alloc ops.
 */
struct ConvertLoomAllocOp : public OpConversionPattern<::loom::AllocOp> {
  using OpConversionPattern<::loom::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::loom::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getResult().use_empty()) {
      return rewriter.notifyMatchFailure(
          op, "loom.alloc still has users; defer erasure until dependent rewrites finish");
    }
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// loom.copy Load/Store Patterns (Reader/Writer Kernels)
//===----------------------------------------------------------------------===//

/**
 * @brief Convert `loom.copy` (load direction) into TTKernel NOC read ops for
 *        reader kernels.
 *
 * @details Handles loom.copy operations where the source is a
 *          memref.reinterpret_cast (data flows from DRAM to L1). Supports
 *          both unicast and broadcast via the op's `interconnect` and
 *          `broadcast` attributes.
 */
struct ConvertLoomMemoryLoadOp : public OpConversionPattern<::loom::CopyOp> {
  using OpConversionPattern<::loom::CopyOp>::OpConversionPattern;

  ConvertLoomMemoryLoadOp(TypeConverter &typeConverter, MLIRContext *context,
                          std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<::loom::CopyOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  LogicalResult
  matchAndRewrite(::loom::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Must be a load: source is a reinterpret_cast.
    Value source = op.getSource();
    auto reinterpretCastOp = source.getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpretCastOp)
      return failure();

    Value inputMemref = reinterpretCastOp.getSource();

    // Get memrefArgData from tracker.
    MemrefArgData *memrefArgData = tracker->getMemrefData(inputMemref);
    if (!memrefArgData) {
      llvm::errs() << "No MemrefArgData found for input memref\n";
      llvm::errs() << "inputMemref: " << inputMemref << "\n";
      return failure();
    }
    Value cb = tracker->getCB(inputMemref);
    if (!cb) {
      llvm::errs() << "Error: CB not found for memref " << inputMemref << "\n";
      return failure();
    }
    auto cbType = cast<CBType>(cb.getType());
    auto elementType = cbType.getElementType();
    Value numPages;

    if (auto tileType = llvm::dyn_cast<TileType>(elementType)) {
      const int32_t numTiles = cbType.getNumTiles();
      numPages = rewriter.create<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), numTiles);
    } else {
      const int64_t numElements = cbType.getNumElements();
      int32_t elementSizeBytes = 4;
      if (elementType.isF16() || elementType.isBF16())
        elementSizeBytes = 2;
      else if (elementType.isF32())
        elementSizeBytes = 4;
      else if (auto intType = llvm::dyn_cast<IntegerType>(elementType))
        elementSizeBytes = (intType.getWidth() + 7) / 8;
      Value pageSize = GetTileSizeOp::create(rewriter, loc, cb);
      Value totalSize = rewriter.create<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), numElements * elementSizeBytes);
      numPages = arith::DivSIOp::create(rewriter, loc, totalSize, pageSize);
    }

    memrefArgData->num_tiles = numPages;
    CBReserveBackOp::create(rewriter, loc, cb, memrefArgData->num_tiles);

    // Determine broadcast from the loom.copy interconnect attribute.
    // interconnect : [@vertical_links]   → isBroadcastY (vertical)
    // interconnect : [@horizontal_links] → isBroadcastX (horizontal)
    // interconnect : []                  → no broadcast (unicast)
    bool isBroadcast = false;
    bool isBroadcastX = false;
    bool isBroadcastAll = false;
    auto interconnectAttr = op.getInterconnect();
    if (interconnectAttr && !interconnectAttr.empty()) {
      isBroadcast = true;
      bool hasHorizontalLinks = false;
      bool hasVerticalLinks = false;
      for (Attribute attr : interconnectAttr) {
        if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr)) {
          StringRef name = symRef.getValue();
          if (name == "horizontal_links") {
            isBroadcastX = true;
            hasHorizontalLinks = true;
          } else if (name == "vertical_links") {
            hasVerticalLinks = true;
          }
        }
      }
      if (hasHorizontalLinks && hasVerticalLinks) {
        isBroadcastAll = true;
      }
    }

    if (isBroadcast) {
      Operation *parentFunc = op->getParentOfType<func::FuncOp>();
      Value coreX;
      Value coreY;
/*       if (failed(generate_multicast_address(
              isBroadcastX, rewriter, loc, memrefArgData, tracker, parentFunc,
              coreX, coreY))) {
        return failure();
      } */

      coreX = tracker->getCoreCoordForDim(parentFunc, "x");
      coreY = tracker->getCoreCoordForDim(parentFunc, "y");
      if (!coreX || !coreY) {
        llvm::errs() << "missing mapped spatial core coordinates for broadcast; "
                        "expected loom.mapped_to_dims to define both @x and @y\n";
        return failure();
      }
      auto toI32 = [&](Value v) -> Value {
        if (v.getType().isIndex())
          return rewriter.create<arith::IndexCastOp>(loc,
                                                     rewriter.getI32Type(), v);
        return v;
      };
  
      coreX = toI32(coreX);
      coreY = toI32(coreY);

      Value zero = rewriter.create<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), 0);
      Value cond;
      if (isBroadcastAll) {
        Value condX = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, coreX, zero);
        Value condY = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, coreY, zero);
        cond = rewriter.create<arith::AndIOp>(loc, condX, condY);
      } else if (isBroadcastX)
        cond = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, coreX, zero);
      else
        cond = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, coreY, zero);

      auto ifOp = rewriter.create<scf::IfOp>(loc, cond, true);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(
            ifOp.getThenRegion().front().getTerminator());
        auto [totalSizeBytes, multicast_l1Addr] = dram_read(
            source, loc, rewriter, memrefArgData, tracker, getTypeConverter());
        if (!totalSizeBytes || !multicast_l1Addr) {
          llvm::errs() << "dram_read failed to return valid values\n";
          return failure();
        }
        multicast_send(rewriter, loc, memrefArgData, totalSizeBytes,
                       multicast_l1Addr);

        rewriter.setInsertionPoint(
            ifOp.getElseRegion().front().getTerminator());
        multicast_receive(rewriter, loc, memrefArgData);
      }
      rewriter.setInsertionPointAfter(ifOp);
    } else {
      auto [totalSizeBytes, multicast_l1Addr] = dram_read(
          source, loc, rewriter, memrefArgData, tracker, getTypeConverter());
      if (!totalSizeBytes || !multicast_l1Addr) {
        llvm::errs() << "dram_read failed to return valid values\n";
        return failure();
      }
    }

    CBPushBackOp::create(rewriter, loc, memrefArgData->cb,
                         memrefArgData->num_tiles);
    rewriter.eraseOp(op);
    return success();
  }

private:
  std::shared_ptr<CompileArgTracker> tracker;
};

/**
 * @brief Convert `loom.copy` (store direction) into TTKernel NOC write ops
 *        for writer kernels.
 *
 * @details Handles loom.copy operations where the destination is a
 *          memref.reinterpret_cast (data flows from L1 to DRAM).
 */
struct ConvertLoomMemoryStoreOp : public OpConversionPattern<::loom::CopyOp> {
  using OpConversionPattern<::loom::CopyOp>::OpConversionPattern;

  ConvertLoomMemoryStoreOp(TypeConverter &typeConverter, MLIRContext *context,
                           std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<::loom::CopyOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  LogicalResult
  matchAndRewrite(::loom::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Store: destination is a reinterpret_cast.
    Value target = op.getDestination();
    auto reinterpretCastOp = target.getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpretCastOp)
      return failure();
    Location loc = op.getLoc();

    Value outputMemref = reinterpretCastOp.getSource();

    Value cb = tracker->getCB(outputMemref);
    if (!cb) {
      llvm::errs() << "No CB found for LoomCopyOp store outputMemref: "
                   << outputMemref << "\n";
      return failure();
    }

    Value baseAddr = tracker->getBaseAddr(outputMemref);
    if (!baseAddr) {
      llvm::errs() << "No base address found for LoomCopyOp store\n";
      return failure();
    }

    // Determine insertion point after both cb and baseAddr.
    Value insertionAnchor = cb;
    Operation *cbOp = cb.getDefiningOp();
    Operation *baseAddrOp = baseAddr.getDefiningOp();
    if (cbOp && baseAddrOp && cbOp->getBlock() == baseAddrOp->getBlock()) {
      if (cbOp->isBeforeInBlock(baseAddrOp))
        insertionAnchor = baseAddr;
    }

    auto opInsertionPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(insertionAnchor);
    auto pageSize = GetTileSizeOp::create(rewriter, loc, cb);

    // Get offset from reinterpret_cast.
    Value offset;
    {
      auto offsets = reinterpretCastOp.getOffsets();
      if (!offsets.empty()) {
        offset = offsets[0];
      } else {
        auto mixedOffsets = reinterpretCastOp.getMixedOffsets();
        if (!mixedOffsets.empty() && isa<Attribute>(mixedOffsets[0])) {
          auto staticOffset =
              llvm::cast<IntegerAttr>(cast<Attribute>(mixedOffsets[0]));
          offset =
              rewriter.create<arith::ConstantIndexOp>(loc, staticOffset.getInt());
        } else {
          offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        }
      }
    }

    rewriter.restoreInsertionPoint(opInsertionPt);

    Value offsetI32 = offset;
    if (offsetI32 && offsetI32.getType().isIndex())
      offsetI32 = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), offsetI32);

    // Compute number of pages.
    auto cbType = cast<CBType>(cb.getType());
    Value numPages;
    auto elementType = cbType.getElementType();
    if (auto tileType = llvm::dyn_cast<TileType>(elementType)) {
      const int32_t numTiles = cbType.getNumTiles();
      numPages = rewriter.create<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), numTiles);
    } else {
      const int64_t numElements = cbType.getNumElements();
      int32_t elementSizeBytes = 4;
      if (elementType.isF16() || elementType.isBF16())
        elementSizeBytes = 2;
      else if (elementType.isF32())
        elementSizeBytes = 4;
      else if (auto intType = llvm::dyn_cast<IntegerType>(elementType))
        elementSizeBytes = (intType.getWidth() + 7) / 8;
      Value totalSizeBytes = rewriter.create<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), numElements * elementSizeBytes);
      numPages =
          arith::DivSIOp::create(rewriter, loc, totalSizeBytes, pageSize);
    }

    Value accessorOp = tracker->getTensorAccessor(outputMemref);
    if (!accessorOp) {
      llvm::errs() << "No TensorAccessor found for output memref\n";
      return failure();
    }

    CBWaitFrontOp::create(rewriter, loc, cb, numPages);
    Value l1Addr = GetReadPtrOp::create(rewriter, loc, cb);

    Value baseElemOffset = offsetI32;
    auto resultType = cast<MemRefType>(reinterpretCastOp.getResult().getType());
    ArrayRef<int64_t> shape = resultType.getShape();

    SmallVector<int64_t> strides;
    auto layout = resultType.getLayout();
    if (auto stridedLayout = dyn_cast<StridedLayoutAttr>(layout)) {
      auto stridesRef = stridedLayout.getStrides();
      strides.append(stridesRef.begin(), stridesRef.end());
    } else {
      int64_t stride = 1;
      for (int i = shape.size() - 1; i >= 0; --i) {
        strides.insert(strides.begin(), stride);
        stride *= shape[i];
      }
    }

    constexpr int64_t kTileDim = 32;
    int64_t numTileRows =
        (shape.size() > 0) ? (shape[shape.size() - 2] + kTileDim - 1) / kTileDim : 1;
    int64_t numTileCols =
        (shape.size() > 1) ? (shape[shape.size() - 1] + kTileDim - 1) / kTileDim : 1;

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
      loc, rewriter.getI32Type(), (strides.size() > 0) ? strides[strides.size() - 2] : 1);
    Value stride1Val = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), (strides.size() > 1) ? strides[strides.size() - 1] : 1);

    scf::ForOp rowLoop =
        scf::ForOp::create(rewriter, loc, loopConst0, numTileRowsVal,
                           loopConst1, ValueRange{l1Addr});
    {
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      Value tileRow = rowLoop.getInductionVar();
      Value crtL1Addr = rowLoop.getRegionIterArgs()[0];

      scf::ForOp colLoop =
          scf::ForOp::create(rewriter, loc, loopConst0, numTileColsVal,
                             loopConst1, ValueRange{crtL1Addr});
      {
        rewriter.setInsertionPointToStart(colLoop.getBody());
        Value tileCol = colLoop.getInductionVar();
        Value innerL1Addr = colLoop.getRegionIterArgs()[0];

        Value rowOffset =
            arith::MulIOp::create(rewriter, loc, tileRow, tileDimVal,
                                  arith::IntegerOverflowFlags::nsw);
        rowOffset = arith::MulIOp::create(rewriter, loc, rowOffset, stride0Val,
                                          arith::IntegerOverflowFlags::nsw);
        Value colOffset =
            arith::MulIOp::create(rewriter, loc, tileCol, tileDimVal,
                                  arith::IntegerOverflowFlags::nsw);
        colOffset = arith::MulIOp::create(rewriter, loc, colOffset, stride1Val,
                                          arith::IntegerOverflowFlags::nsw);
        Value tileElemOffset =
            arith::AddIOp::create(rewriter, loc, rowOffset, colOffset,
                                  arith::IntegerOverflowFlags::nsw);
        Value totalElemOffset = arith::AddIOp::create(
            rewriter, loc, baseElemOffset, tileElemOffset,
            arith::IntegerOverflowFlags::nsw);

        Value rowIdx =
            arith::DivSIOp::create(rewriter, loc, totalElemOffset, stride0Val);
        Value colIdx =
            arith::RemSIOp::create(rewriter, loc, totalElemOffset, stride0Val);
        Value tilesPerRow =
            arith::DivSIOp::create(rewriter, loc, stride0Val, tileDimVal);
        Value rowTile =
            arith::DivSIOp::create(rewriter, loc, rowIdx, tileDimVal);
        Value rowTileBase =
            arith::MulIOp::create(rewriter, loc, rowTile, tilesPerRow,
                                  arith::IntegerOverflowFlags::nsw);
        Value colTile =
            arith::DivSIOp::create(rewriter, loc, colIdx, tileDimVal);
        Value tileId =
            arith::AddIOp::create(rewriter, loc, rowTileBase, colTile,
                                  arith::IntegerOverflowFlags::nsw);
        NocAsyncWriteTileOp::create(rewriter, loc, tileId, accessorOp,
                                    innerL1Addr);

        Value nextL1Addr =
            arith::AddIOp::create(rewriter, loc, innerL1Addr, pageSize,
                                  arith::IntegerOverflowFlags::nsw);
        scf::YieldOp::create(rewriter, loc, ValueRange(nextL1Addr));
      }

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
  std::shared_ptr<CompileArgTracker> tracker;
};

//===----------------------------------------------------------------------===//
// loom.copy Compute Kernel Patterns
//===----------------------------------------------------------------------===//

/**
 * @brief Convert `loom.copy` (load) in compute kernels.
 *
 * @details Compute-side CB synchronization is handled by compute op lowering
 *          (e.g. matmul/generic patterns). The load op is erased here.
 */
struct ConvertLoomComputeLoadOp : public OpConversionPattern<::loom::CopyOp> {
  using OpConversionPattern<::loom::CopyOp>::OpConversionPattern;

  ConvertLoomComputeLoadOp(TypeConverter &typeConverter, MLIRContext *context,
                           std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<::loom::CopyOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  LogicalResult
  matchAndRewrite(::loom::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }

private:
  std::shared_ptr<CompileArgTracker> tracker;
};

/**
 * @brief Convert `loom.copy` (store) to CB synchronization for compute
 *        kernels.
 *
 * @details In compute kernels, matmul lowering materializes tile register
 *          commit/wait and pack operations. Store lowering only emits
 *          cb_push_back so writer kernels can drain the CB to DRAM.
 */
struct ConvertLoomComputeStoreOp : public OpConversionPattern<::loom::CopyOp> {
  using OpConversionPattern<::loom::CopyOp>::OpConversionPattern;

  ConvertLoomComputeStoreOp(TypeConverter &typeConverter, MLIRContext *context,
                            std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<::loom::CopyOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  LogicalResult
  matchAndRewrite(::loom::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Store: destination is a reinterpret_cast.
    Value target = op.getDestination();
    auto reinterpretCastOp = target.getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpretCastOp)
      return failure();

    Value outputMemref = reinterpretCastOp.getSource();

    Value outcb = tracker->getCB(outputMemref);
    if (!outcb || !isa<CBType>(outcb.getType())) {
      llvm::errs() << "No CB found for LoomComputeStoreOp target\n";
      return failure();
    }

    auto outcbType = cast<CBType>(outcb.getType());
    int32_t numTiles = static_cast<int32_t>(outcbType.getNumElements()) / 1024;
    (void)numTiles;

    if (auto pairedGive = findAdjacentSemaphoreGiveAfterStore(op))
      rewriter.eraseOp(pairedGive);

    // CBPushBack stays disabled for now.
    rewriter.eraseOp(op);
    return success();
  }

private:
  std::shared_ptr<CompileArgTracker> tracker;
};

//===----------------------------------------------------------------------===//
// loom.copy Dispatchers
//===----------------------------------------------------------------------===//

/**
 * @brief Dispatcher pattern for loom.copy load operations.
 *
 * @details Routes loom.copy (load) operations to the appropriate pattern
 *          based on kernel type:
 *          - Compute kernels: delegates to ConvertLoomComputeLoadOp
 *          - Memory kernels (reader): delegates to ConvertLoomMemoryLoadOp
 */
struct ConvertLoomLoadOp : public OpConversionPattern<::loom::CopyOp> {
  using OpConversionPattern<::loom::CopyOp>::OpConversionPattern;

  ConvertLoomLoadOp(TypeConverter &typeConverter, MLIRContext *context,
                    std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<::loom::CopyOp>(typeConverter, context),
        computePattern(typeConverter, context, tracker),
        memoryPattern(typeConverter, context, tracker) {}

  LogicalResult
  matchAndRewrite(::loom::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Must be a load: source is a reinterpret_cast.
    Value source = op.getSource();
    if (!source.getDefiningOp<memref::ReinterpretCastOp>())
      return failure();

    if (isComputeKernel(op))
      return computePattern.matchAndRewrite(op, adaptor, rewriter);
    return memoryPattern.matchAndRewrite(op, adaptor, rewriter);
  }

private:
  ConvertLoomComputeLoadOp computePattern;
  ConvertLoomMemoryLoadOp memoryPattern;
};

/**
 * @brief Dispatcher pattern for loom.copy store operations.
 *
 * @details Routes loom.copy (store) operations to the appropriate pattern
 *          based on kernel type:
 *          - Compute kernels: delegates to ConvertLoomComputeStoreOp
 *          - Memory kernels (writer): delegates to ConvertLoomMemoryStoreOp
 */
struct ConvertLoomStoreOp : public OpConversionPattern<::loom::CopyOp> {
  using OpConversionPattern<::loom::CopyOp>::OpConversionPattern;

  ConvertLoomStoreOp(TypeConverter &typeConverter, MLIRContext *context,
                     std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<::loom::CopyOp>(typeConverter, context),
        computePattern(typeConverter, context, tracker),
        memoryPattern(typeConverter, context, tracker) {}

  LogicalResult
  matchAndRewrite(::loom::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Must be a store: destination is a reinterpret_cast.
    Value target = op.getDestination();
    if (!target.getDefiningOp<memref::ReinterpretCastOp>())
      return failure();

    if (isComputeKernel(op))
      return computePattern.matchAndRewrite(op, adaptor, rewriter);
    return memoryPattern.matchAndRewrite(op, adaptor, rewriter);
  }

private:
  ConvertLoomComputeStoreOp computePattern;
  ConvertLoomMemoryStoreOp memoryPattern;
};

void mlir::loom::populateMemoryOpConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context, std::shared_ptr<CompileArgTracker> tracker) {
  // loom.semaphore / loom.copy patterns.
  patterns.add<ConvertLoomSemaphoreTakeOp>(typeConverter, context, tracker);
  patterns.add<ConvertLoomSemaphoreGiveOp>(typeConverter, context);
  patterns.add<ConvertLoomLoadOp>(typeConverter, context, tracker);
  patterns.add<ConvertLoomStoreOp>(typeConverter, context, tracker);
  // Reinterpret cast erasure.
  patterns.add<ConvertReinterpretCastOp>(typeConverter, context);
}

void mlir::loom::populateLoomAllocCleanupPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context) {
  patterns.add<ConvertLoomAllocOp>(typeConverter, context);
}
