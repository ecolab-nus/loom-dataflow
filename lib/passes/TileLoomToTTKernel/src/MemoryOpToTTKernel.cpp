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
#include "TTKernelAttrs.h"
#include "TTKernelUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
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
#include "LoomEnums.h.inc"
#include "LoomAttributes.h.inc"
#include "LoomOps.h.inc"

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;
using namespace tt::ttcore;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/**
 * @brief Compute semaphore-give tile count from a shaped value.
 *
 * @details
 * Uses tensor rank semantics where the trailing two dimensions map to a tile
 * plane and all leading dimensions are batch multiplicity. The trailing two
 * dims are each clamped to at least 32 so narrow vectors/scalars still occupy
 * one 32x32 tile per leading slice.
 *
 * Examples:
 * - `8x1x32` -> `8 * ceil(max(1,32)/32) * ceil(max(32,32)/32)` = `8`
 * - `8x1x32x128` -> `8 * 1 * 4` = `32`
 *
 * @param type The shaped type backing semaphore-give.
 * @return Number of tiles, or nullopt for dynamic/invalid shapes.
 */
static std::optional<int64_t> getNumTilesForSemaphoreGive(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return std::nullopt;

  ArrayRef<int64_t> shape = shaped.getShape();
  if (shape.empty())
    return int64_t{1};

  int64_t leadingMultiplier = 1;
  if (shape.size() > 2) {
    for (int64_t dim : shape.drop_back(2)) {
      if (dim <= 0)
        return std::nullopt;
      leadingMultiplier *= dim;
    }
  }

  int64_t height = shape.size() >= 2 ? shape[shape.size() - 2] : 1;
  int64_t width = shape.back();
  if (height <= 0 || width <= 0)
    return std::nullopt;

  if (height < 32)
    height = 32;
  if (width < 32)
    width = 32;

  auto heightTiles = ceilDiv32(height);
  auto widthTiles = ceilDiv32(width);
  if (!heightTiles || !widthTiles)
    return std::nullopt;

  return leadingMultiplier * (*heightTiles) * (*widthTiles);
}

/**
 * @brief Compute the number of 32x32 tiles represented by a CB type.
 *
 * @details
 * - For tile-typed CBs, TTKernel stores the tile count directly.
 * - For element-typed CBs (e.g. f16/f32/i32), the CB stores element count and
 *   we convert to tile count via ceil(numElements / 1024).
 *
 * This avoids relying on shaped memref rank semantics (which can undercount
 * stacked gather payloads such as `8x64x64`).
 *
 * @param cbType Circular buffer type.
 * @return Number of tiles, or nullopt for invalid/non-positive sizes.
 */
static std::optional<int64_t> getNumTilesFromCBType(CBType cbType) {
  if (!cbType)
    return std::nullopt;

  Type elementType = cbType.getElementType();
  if (isa<TileType>(elementType)) {
    int64_t tiles = cbType.getNumTiles();
    return tiles > 0 ? std::optional<int64_t>(tiles) : std::nullopt;
  }

  int64_t numElements = cbType.getNumElements();
  if (numElements <= 0)
    return std::nullopt;

  constexpr int64_t kTileElements = 32 * 32;
  return (numElements + kTileElements - 1) / kTileElements;
}

// Emit ceil_div(lhs, rhs) using ops that survive downstream TTKernel->EmitC.
static Value ceilDivSICompat(ConversionPatternRewriter &rewriter, Location loc,
                             Value lhs, Value rhs) {
  Type ty = lhs.getType();
  Value one = arith::ConstantIntOp::create(rewriter, loc, ty, 1);
  Value rhsMinusOne = arith::SubIOp::create(
      rewriter, loc, rhs, one, arith::IntegerOverflowFlags::nsw);
  Value numer = arith::AddIOp::create(rewriter, loc, lhs, rhsMinusOne,
                                      arith::IntegerOverflowFlags::nsw);
  return arith::DivSIOp::create(rewriter, loc, numer, rhs);
}

using ReduceTransportAnalysis = ReduceCoreRegionAnalysis;

struct ReduceTransportRuntimeValues {
  Value payloadCb;
  Value outCb;
  Value numTiles;
  Value payloadBytes;
  Value reducerDestNocX;
  Value reducerDestNocY;
  Value readySemaphorePtr;
  Value tokenSemaphoreAddr;
  Value tokenSemaphorePtr;
  Value tokenSemaphoreMcastNocAddr;
  Value reducerReadySemaphoreNocAddr;
  Value payloadReadPtr;
  Value reducerOutWritePtr;
  Value zero;
  Value one;
  Value nocId;
};

static FailureOr<ReduceTransportAnalysis> analyzeReduceTransportRegion(
    ::loom::GatherOp op, ConversionPatternRewriter &rewriter,
    std::shared_ptr<CompileArgTracker> tracker) {
  if (!tracker)
    return failure();
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return failure();
  Location loc = op.getLoc();

  return analyzeReduceCoreRegion(
      rewriter, loc, tracker->getCoreCoordForDim(parentFunc, "x"),
      tracker->getCoreCoordForDim(parentFunc, "y"), op.getUlX(), op.getUlY(),
      op.getLrX(), op.getLrY());
}

static FailureOr<ReduceTransportRuntimeValues> materializeReduceTransportRuntime(
    ::loom::GatherOp op, Value payloadCb, Value outCb,
    ConversionPatternRewriter &rewriter,
    std::shared_ptr<CompileArgTracker> tracker,
    const ReduceTransportAnalysis &analysis) {
  if (!tracker)
    return failure();
  if (!isa<CBType>(payloadCb.getType()) || !isa<CBType>(outCb.getType()))
    return failure();

  auto numTilesOpt = getNumTilesFromShapedType(op.getIns().getType());
  if (!numTilesOpt)
    return failure();
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return failure();
  auto *reduceRuntimeArgs =
      tracker->getReduceRuntimeArgs(parentFunc.getOperation());
  if (!reduceRuntimeArgs)
    return failure();

  Location loc = op.getLoc();
  Value zero = i32Const(rewriter, loc, 0);
  Value one = i32Const(rewriter, loc, 1);
  int8_t writerNocId =
      getDataMovementKernelSpec(DataMovementKernelRole::Writer).nocId;
  Value nocId = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI8Type(), writerNocId);
  Value numTiles = i32Const(rewriter, loc, *numTilesOpt);
  Value payloadBytes = arith::MulIOp::create(
      rewriter, loc, numTiles, GetTileSizeOp::create(rewriter, loc, payloadCb));

  Value readySemaphoreAddr =
      GetSemaphoreOp::create(rewriter, loc, reduceRuntimeArgs->readySemaphore);
  Value tokenSemaphoreAddr =
      GetSemaphoreOp::create(rewriter, loc, reduceRuntimeArgs->tokenSemaphore);
  Value readySemaphorePtr =
      CastToL1PtrOp::create(rewriter, loc, readySemaphoreAddr);
  Value tokenSemaphorePtr =
      CastToL1PtrOp::create(rewriter, loc, tokenSemaphoreAddr);
  Value tokenSemaphoreMcastNocAddr =
      rewriter.create<ExperimentalGetNocMulticastAddrOp>(
                  loc, reduceRuntimeArgs->tokenSemaphoreMcastDestStartX,
                  reduceRuntimeArgs->tokenSemaphoreMcastDestStartY,
                  reduceRuntimeArgs->tokenSemaphoreMcastDestEndX,
                  reduceRuntimeArgs->tokenSemaphoreMcastDestEndY,
                  tokenSemaphoreAddr, nocId)
          .getResult();
  // GetNocAddrOp expects physical NOC coordinates; reuse reducer destination
  // physical coords provided by reduce runtime args.
  Value reducerDestNocX = reduceRuntimeArgs->tokenSemaphoreMcastDestStartX;
  Value reducerDestNocY = reduceRuntimeArgs->tokenSemaphoreMcastDestStartY;
  Value reducerReadySemaphoreNocAddr =
      GetNocAddrOp::create(rewriter, loc, reducerDestNocX, reducerDestNocY,
                           readySemaphoreAddr);

  return ReduceTransportRuntimeValues{
      payloadCb,
      outCb,
      numTiles,
      payloadBytes,
      reducerDestNocX,
      reducerDestNocY,
      readySemaphorePtr,
      tokenSemaphoreAddr,
      tokenSemaphorePtr,
      tokenSemaphoreMcastNocAddr,
      reducerReadySemaphoreNocAddr,
      GetReadPtrOp::create(rewriter, loc, payloadCb),
      GetWritePtrOp::create(rewriter, loc, outCb),
      zero,
      one,
      nocId};
}

static void emitWorkerReduceTransport(ConversionPatternRewriter &rewriter,
                                      Location loc,
                                      const ReduceTransportAnalysis &analysis,
                                      const ReduceTransportRuntimeValues &runtime,
                                      ReduceProtocol protocol) {
  CBWaitFrontOp::create(rewriter, loc, runtime.payloadCb, runtime.numTiles);
  if (protocol == ReduceProtocol::MultiSlot) {
    NocSemaphoreWaitOp::create(rewriter, loc, runtime.tokenSemaphorePtr,
                               runtime.one);
  } else {
    NocSemaphoreWaitOp::create(rewriter, loc, runtime.tokenSemaphorePtr,
                               analysis.rank);
  }

  // Gather always writes rank-indexed slots in outCb. Single-slot only
  // changes token synchronization ordering.
  Value reducerWriteBase = runtime.reducerOutWritePtr;
  Value slot = analysis.rank;
  Value slotOffsetBytes =
      arith::MulIOp::create(rewriter, loc, slot, runtime.payloadBytes);
  Value reducerSlotL1 = arith::AddIOp::create(
      rewriter, loc, reducerWriteBase, slotOffsetBytes);
  Value reducerSlotNocAddr = GetNocAddrOp::create(
      rewriter, loc, runtime.reducerDestNocX, runtime.reducerDestNocY,
      reducerSlotL1);

  NocAsyncWriteOp::create(rewriter, loc, runtime.payloadReadPtr,
                          reducerSlotNocAddr, runtime.payloadBytes);
  NocAsyncWriteBarrierOp::create(rewriter, loc);
  //reset the token semaphore to 0
  NocSemaphoreSetOp::create(rewriter, loc, runtime.tokenSemaphorePtr, runtime.zero);
  NocSemaphoreIncOp::create(rewriter, loc, runtime.reducerReadySemaphoreNocAddr,
                            runtime.one, runtime.nocId);
  CBPopFrontOp::create(rewriter, loc, runtime.payloadCb, runtime.numTiles);
}

static void emitReducerReduceTransportSync(
    ConversionPatternRewriter &rewriter, Location loc,
    const ReduceTransportAnalysis &analysis,
    const ReduceTransportRuntimeValues &runtime, ReduceProtocol protocol) {
  auto falseAttr = rewriter.getBoolAttr(false);
  Value allParticipantTiles = arith::MulIOp::create(
      rewriter, loc, analysis.participants, runtime.numTiles);
  //reserve space for remote workers
  CBReserveBackOp::create(rewriter, loc, runtime.outCb, allParticipantTiles);
  //reducer ready to receive payload
  NocSemaphoreSetOp::create(rewriter, loc, runtime.tokenSemaphorePtr, runtime.one);
  NocSemaphoreSetMulticastOp::create(
    rewriter, loc, runtime.tokenSemaphoreAddr, runtime.tokenSemaphoreMcastNocAddr,
    analysis.workerCount, falseAttr, falseAttr);

  // Place reducer-local payload at rank-0 slot of outCb.
  CBWaitFrontOp::create(rewriter, loc, runtime.payloadCb, runtime.numTiles);
  Value reducerSlotNocAddr = GetNocAddrOp::create(
      rewriter, loc, runtime.reducerDestNocX, runtime.reducerDestNocY,
      runtime.reducerOutWritePtr);
  NocAsyncWriteOp::create(rewriter, loc, runtime.payloadReadPtr, reducerSlotNocAddr,
                          runtime.payloadBytes);
  NocAsyncWriteBarrierOp::create(rewriter, loc);
  CBPopFrontOp::create(rewriter, loc, runtime.payloadCb, runtime.numTiles);

  if (protocol == ReduceProtocol::MultiSlot) {
    NocSemaphoreWaitOp::create(rewriter, loc, runtime.readySemaphorePtr,
                               analysis.workerCount);
    NocSemaphoreSetOp::create(rewriter, loc, runtime.readySemaphorePtr, runtime.zero);
    CBPushBackOp::create(rewriter, loc, runtime.outCb, allParticipantTiles);
    return;
  }
  // In single-slot mode, reducer token-serializes workers by rank.
  scf::ForOp workerLoop = scf::ForOp::create(
      rewriter, loc, runtime.one, analysis.participants, runtime.one);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(workerLoop.getBody());
    Value rank = workerLoop.getInductionVar();
    NocSemaphoreSetOp::create(rewriter, loc, runtime.tokenSemaphorePtr, rank);
    NocSemaphoreSetMulticastOp::create(
      rewriter, loc, runtime.tokenSemaphoreAddr, runtime.tokenSemaphoreMcastNocAddr,
      analysis.workerCount, falseAttr, falseAttr);
    NocSemaphoreWaitOp::create(rewriter, loc, runtime.readySemaphorePtr, rank);
  }
  CBPushBackOp::create(rewriter, loc, runtime.outCb, allParticipantTiles);
}

static std::optional<int64_t> getAnnotatedVecTiles(::loom::CopyOp op) {
  auto tilesAttr = op->getAttrOfType<IntegerAttr>(kVecTilesAttrName);
  if (!tilesAttr)
    return std::nullopt;
  int64_t tiles = tilesAttr.getInt();
  if (tiles <= 0)
    return std::nullopt;
  return tiles;
}

static std::optional<int64_t> getRank1VecTilesFromMemref(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType || !memrefType.hasStaticShape() || memrefType.getRank() != 1)
    return std::nullopt;
  return ceilDiv32(memrefType.getShape().front());
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
 * @brief Strip lightweight cast wrappers introduced during conversion.
 *
 * @details Removes:
 * - `memref.cast`
 * - single-input `builtin.unrealized_conversion_cast`
 */
static Value stripConversionCasts(Value value) {
  Value current = value;
  while (true) {
    if (auto cast = current.getDefiningOp<memref::CastOp>()) {
      current = cast.getSource();
      continue;
    }
    if (auto unrealized =
            current.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (unrealized.getNumOperands() == 1) {
        current = unrealized.getOperand(0);
        continue;
      }
    }
    break;
  }
  return current;
}

/**
 * @brief Recover the base semaphore-backed memref from a rewritten operand.
 *
 * @details Walks through conversion casts and nested `loom.semaphore_take`
 * wrappers to reach the underlying memref value whenever possible.
 */
static Value resolveSemaphoreSourceMemref(Value value) {
  Value current = stripConversionCasts(value);
  while (auto semTake = current.getDefiningOp<::loom::SemaphoreTakeOp>())
    current = stripConversionCasts(semTake.getSource());
  return current;
}

static SmallVector<int64_t, 2> collectCopyBindingSlotsForEndpoint(Value endpoint) {
  SmallVector<int64_t, 2> slots;
  if (!endpoint)
    return slots;

  for (Operation *user : endpoint.getUsers()) {
    auto copyOp = dyn_cast<::loom::CopyOp>(user);
    if (!copyOp)
      continue;
    if (copyOp.getSource() != endpoint && copyOp.getDestination() != endpoint)
      continue;
    auto slot = getCopyBindingSlot(copyOp.getOperation());
    if (!slot)
      continue;
    if (!llvm::is_contained(slots, *slot))
      slots.push_back(*slot);
  }

  return slots;
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
  if (!isL1ToDramCopy(copyOp))
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
  if (!copyOp || !isL1ToDramCopy(copyOp))
    return false;

  return stripMemrefCasts(copyOp.getSource()) ==
         stripMemrefCasts(giveOp.getSource());
}

/**
 * @brief Check whether semaphore_give source aliases a loom.gather input.
 *
 * @details If a buffer is consumed as `loom.gather` input, gather transport
 *          owns the source-consumption semantics. Lowering an additional
 *          `semaphore_give` into `cb_pop_front` would double-consume pages.
 */
static bool isSemaphoreGiveForGatherInput(::loom::SemaphoreGiveOp giveOp) {
  auto parentFunc = giveOp->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return false;

  Value giveBase = resolveSemaphoreSourceMemref(giveOp.getSource());
  if (!giveBase)
    return false;

  bool found = false;
  parentFunc.walk([&](::loom::GatherOp gatherOp) {
    if (found)
      return;
    Value gatherInsBase = resolveSemaphoreSourceMemref(gatherOp.getIns());
    found = gatherInsBase && gatherInsBase == giveBase;
  });

  return found;
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
 * @param bindingData       Pre-computed per-copy runtime tuple metadata.
 * @return A pair of (totalSizeBytes, multicast_l1Addr), or (null, null) on
 *         error.
 */
std::pair<Value, Value> dram_read(Value source, Location loc,
                                   ConversionPatternRewriter &rewriter,
                                   CopyBindingData *bindingData) {
  auto reinterpretCastOp = source.getDefiningOp<memref::ReinterpretCastOp>();
  if (!reinterpretCastOp) {
    emitError(loc) << "DRAM read source must be memref.reinterpret_cast";
    return std::make_pair(Value(), Value());
  }

  if (!bindingData) {
    emitError(loc) << "missing runtime tuple for DRAM read copy binding";
    return std::make_pair(Value(), Value());
  }

  Value cb = bindingData->cb;
  if (!cb) {
    emitError(loc) << "missing CB runtime arg for DRAM read copy binding";
    return std::make_pair(Value(), Value());
  }


  Value baseAddr = bindingData->baseAddr;
  if (!baseAddr) {
    emitError(loc) << "missing base address runtime arg for DRAM read copy binding";
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
        emitError(loc) << "memref.reinterpret_cast for DRAM read must carry "
                          "a dynamic or static offset";
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

  // Use num_tiles from bindingData (pre-computed by caller for domination correctness)
  // If not set yet (non-broadcast case), compute it now.
  Value numPages = bindingData->num_tiles;
  Value totalSizeBytes = arith::MulIOp::create(rewriter, loc, numPages, pageSize);

  Value accessorOp = bindingData->tensorAccessor;
  if (!accessorOp) {
    emitError(loc) << "missing TensorAccessor runtime arg for DRAM read copy "
                      "binding";
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
    emitError(loc) << "DRAM read requires explicit strided memref layout; "
                      "row-major stride synthesis is not allowed";
    return std::make_pair(Value(), Value());
  }

  // Tile dimension (32x32 for TTKernel).
  constexpr int64_t kTileDim = 32;

  // Create constants for loop bounds and calculations.
  Value loopConst0 =
      rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
  Value loopConst1 =
      rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 1);
  Value tileDimVal = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), kTileDim);

  if (shape.size() <= 1) {
    // Rank-0/1 payloads use a synthetic [1, N] view so tile-id mapping
    // remains valid and avoids rank-2 indexing assumptions.
    int64_t logicalCols = shape.empty() ? 1 : shape.back();
    int64_t numTileCols = (logicalCols + kTileDim - 1) / kTileDim;

    Value numTileColsVal = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), numTileCols);
    Value logicalColsVal = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), logicalCols);

    scf::ForOp colLoop = scf::ForOp::create(rewriter, loc, loopConst0,
                                            numTileColsVal, loopConst1,
                                            ValueRange{l1Addr});
    {
      rewriter.setInsertionPointToStart(colLoop.getBody());
      Value tileCol = colLoop.getInductionVar();
      Value innerL1Addr = colLoop.getRegionIterArgs()[0];

      Value tileElemOffset = arith::MulIOp::create(
          rewriter, loc, tileCol, tileDimVal, arith::IntegerOverflowFlags::nsw);
      Value totalElemOffset = arith::AddIOp::create(
          rewriter, loc, baseElemOffset, tileElemOffset,
          arith::IntegerOverflowFlags::nsw);

      Value rowIdx =
          arith::DivSIOp::create(rewriter, loc, totalElemOffset, logicalColsVal);
      Value colIdx =
          arith::RemSIOp::create(rewriter, loc, totalElemOffset, logicalColsVal);
      Value rowTile = arith::DivSIOp::create(rewriter, loc, rowIdx, tileDimVal);
      Value rowTileBase = arith::MulIOp::create(
          rewriter, loc, rowTile, numTileColsVal,
          arith::IntegerOverflowFlags::nsw);
      Value colTile = arith::DivSIOp::create(rewriter, loc, colIdx, tileDimVal);
      Value tileId = arith::AddIOp::create(rewriter, loc, rowTileBase, colTile,
                                           arith::IntegerOverflowFlags::nsw);

      NocAsyncReadTileOp::create(rewriter, loc, tileId, accessorOp, innerL1Addr);
      Value nextL1Addr = arith::AddIOp::create(
          rewriter, loc, innerL1Addr, pageSize, arith::IntegerOverflowFlags::nsw);
      scf::YieldOp::create(rewriter, loc, ValueRange(nextL1Addr));
    }
    rewriter.setInsertionPointAfter(colLoop);
  } else {
    // Calculate number of tiles in each dimension: numTiles = shape / 32.
    int64_t numTileRows = (shape[shape.size() - 2] + kTileDim - 1) / kTileDim;
    int64_t numTileCols = (shape[shape.size() - 1] + kTileDim - 1) / kTileDim;

    Value numTileRowsVal = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), numTileRows);
    Value numTileColsVal = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), numTileCols);
    Value stride0Val = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(),
        (strides.size() > 1) ? strides[strides.size() - 2] : 1);
    Value stride1Val = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(),
        (strides.size() > 0) ? strides[strides.size() - 1] : 1);

    int64_t totalTiles = numTileRows * numTileCols;
    bool enableReadBarrier = totalTiles > 16;
    int64_t numCores =
        bindingData->parallelCoreCount > 0 ? bindingData->parallelCoreCount : 1;
    int64_t barrierReadThreshold = 1;
    Value barrierCount =
        rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
    if (enableReadBarrier) {
      // 2048 represents the tile size.
      barrierReadThreshold = (512 / numCores) * (1024 + 128) / 2048;
      if (barrierReadThreshold < 1)
        barrierReadThreshold = 1;
    }

    // Create nested loops to iterate over all tiles.
    scf::ForOp rowLoop = scf::ForOp::create(rewriter, loc, loopConst0,
                                            numTileRowsVal, loopConst1,
                                            ValueRange{l1Addr, barrierCount});
    {
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      Value tileRow = rowLoop.getInductionVar();
      Value crtL1Addr = rowLoop.getRegionIterArgs()[0];
      barrierCount = rowLoop.getRegionIterArgs()[1];

      scf::ForOp colLoop = scf::ForOp::create(rewriter, loc, loopConst0,
                                              numTileColsVal, loopConst1,
                                              ValueRange{crtL1Addr,
                                                         barrierCount});
      {
        rewriter.setInsertionPointToStart(colLoop.getBody());
        Value tileCol = colLoop.getInductionVar();
        Value innerL1Addr = colLoop.getRegionIterArgs()[0];
        barrierCount = colLoop.getRegionIterArgs()[1];

        Value rowOffset = arith::MulIOp::create(
            rewriter, loc, tileRow, tileDimVal, arith::IntegerOverflowFlags::nsw);
        rowOffset = arith::MulIOp::create(rewriter, loc, rowOffset, stride0Val,
                                          arith::IntegerOverflowFlags::nsw);

        Value colOffset = arith::MulIOp::create(
            rewriter, loc, tileCol, tileDimVal, arith::IntegerOverflowFlags::nsw);
        colOffset = arith::MulIOp::create(rewriter, loc, colOffset, stride1Val,
                                          arith::IntegerOverflowFlags::nsw);

        Value tileElemOffset = arith::AddIOp::create(
            rewriter, loc, rowOffset, colOffset, arith::IntegerOverflowFlags::nsw);
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
        Value rowTileBase = arith::MulIOp::create(
            rewriter, loc, rowTile, tilesPerRow, arith::IntegerOverflowFlags::nsw);
        Value colTile =
            arith::DivSIOp::create(rewriter, loc, colIdx, tileDimVal);
        Value tileId = arith::AddIOp::create(rewriter, loc, rowTileBase, colTile,
                                             arith::IntegerOverflowFlags::nsw);

        NocAsyncReadTileOp::create(rewriter, loc, tileId, accessorOp, innerL1Addr);
        Value nextL1Addr = arith::AddIOp::create(
            rewriter, loc, innerL1Addr, pageSize, arith::IntegerOverflowFlags::nsw);
        if (enableReadBarrier) {
          barrierCount = arith::AddIOp::create(
              rewriter, loc, barrierCount, loopConst1,
              arith::IntegerOverflowFlags::nsw);
          Value barrierThreshold = rewriter.create<arith::ConstantIntOp>(
              loc, rewriter.getI32Type(), barrierReadThreshold);
          Value shouldReadBarrier = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq,
              barrierCount, barrierThreshold);

          auto barrierIf = scf::IfOp::create(
              rewriter, loc, TypeRange{rewriter.getI32Type()},
              shouldReadBarrier, true);
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(
                &barrierIf.getThenRegion().front());
            NocAsyncReadBarrierOp::create(rewriter, loc);
            scf::YieldOp::create(rewriter, loc, ValueRange{loopConst0});

            rewriter.setInsertionPointToStart(
                &barrierIf.getElseRegion().front());
            scf::YieldOp::create(rewriter, loc, ValueRange{barrierCount});
          }
          rewriter.setInsertionPointAfter(barrierIf);
          barrierCount = barrierIf.getResult(0);
        }
        scf::YieldOp::create(rewriter, loc,
                             ValueRange{nextL1Addr, barrierCount});
      }

      rewriter.setInsertionPointAfter(colLoop);
      scf::YieldOp::create(rewriter, loc, colLoop.getResults());
    }
    rewriter.setInsertionPointAfter(rowLoop);
  }

  // Barrier to wait for all async reads to complete.
  NocAsyncReadBarrierOp::create(rewriter, loc);

  return std::make_pair(totalSizeBytes, multicast_l1Addr);
}

bool multicast_send(ConversionPatternRewriter &rewriter, Location loc, MemrefArgData *memrefArgData, Value totalSizeBytes, Value multicast_l1Addr) {
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
  auto falseAttr = rewriter.getBoolAttr(false);
  // Create noc_id as a Value
  Value nocIdVal = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI8Type(), memrefArgData->noc_id);
  Value noc_multicast_addr =
      rewriter.create<ExperimentalGetNocMulticastAddrOp>(
                  loc, memrefArgData->mcast_dest_noc_start_x,
                  memrefArgData->mcast_dest_noc_start_y,
                  memrefArgData->mcast_dest_noc_end_x,
                  memrefArgData->mcast_dest_noc_end_y, multicast_l1Addr,
                  nocIdVal)
          .getResult();

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
  //only work for blackhole arch
  rewriter.create<emitc::VerbatimOp>(loc, "#ifdef ARCH_BLACKHOLE");
  rewriter.create<emitc::VerbatimOp>(loc, "noc_async_writes_flushed();");
  rewriter.create<emitc::VerbatimOp>(loc, "#endif  // ARCH_BLACKHOLE");

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
 *          physical source buffer.
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

  bool isWriterGatherTransportBuffer(::loom::SemaphoreTakeOp semaphore) const {
    if (!isWriterKernel(semaphore))
      return false;

    auto stripSemaphoreAndCasts = [](Value value) -> Value {
      Value current = stripMemrefCasts(value);
      while (auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>())
        current = stripMemrefCasts(sem.getSource());
      return current;
    };

    Value semaphoreBase = stripSemaphoreAndCasts(semaphore.getSource());
    if (!semaphoreBase)
      return false;

    auto parentFunc = semaphore->getParentOfType<func::FuncOp>();
    if (!parentFunc)
      return false;

    bool found = false;
    parentFunc.walk([&](::loom::GatherOp gatherOp) {
      if (found)
        return;
      Value gatherInsBase = stripSemaphoreAndCasts(gatherOp.getIns());
      Value gatherOutsBase = stripSemaphoreAndCasts(gatherOp.getInit());
      found = gatherInsBase == semaphoreBase || gatherOutsBase == semaphoreBase;
    });
    return found;
  }

  LogicalResult
  matchAndRewrite(::loom::SemaphoreTakeOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    bool isReductionScaleCb = op->hasAttr(kReductionScaleCbAttrName);
    bool isWriterKernelOp = isWriterKernel(op);
    bool isDataMovementScaleInitTarget = isReductionScaleCb && isWriterKernelOp;
    bool isWriterGatherTransportTarget = isWriterGatherTransportBuffer(op);

    // semaphore_take participates in CB handle materialization for compute
    // kernels and writer kernels. Writer-side materialization keeps runtime
    // argument indexing aligned with compute/host CB ordering for cross-kernel
    // transport buffers (e.g. gather/reduce payload/output paths).
    // Reader/host kernels keep memref flow so loom.copy rewrites can consume
    // and erase semaphore/copy scaffolding.
    if (!isComputeKernel(op) && !isWriterKernelOp && !isDataMovementScaleInitTarget &&
        !isWriterGatherTransportTarget) {
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
    SmallVector<int64_t, 2> bindingSlots =
        collectCopyBindingSlotsForEndpoint(op.getResult());
    if (bindingSlots.size() > 1) {
      return rewriter.notifyMatchFailure(
          op, "semaphore participates in multiple DRAM/L1 copy bindings");
    }
    if (bindingSlots.size() == 1) {
      auto *bindingData = tracker->getCopyBindingData(parentFunc.getOperation(),
                                                      bindingSlots.front());
      if (!bindingData) {
        return rewriter.notifyMatchFailure(
            op, "missing copy binding runtime tuple for semaphore");
      }
      cb = bindingData->cb;
    }
    if (!cb) {
      if (auto slotAttr =
              op->getAttrOfType<IntegerAttr>(kCopyBindingSlotAttrName)) {
        auto *bindingData =
            tracker->getCopyBindingData(parentFunc.getOperation(),
                                        slotAttr.getInt());
        if (!bindingData)
          return op.emitOpError()
                 << "missing copy binding runtime tuple for semaphore slot "
                 << slotAttr.getInt();
        cb = bindingData->cb;
      }
    }

    if (!cb) {
      if (auto slotAttr = op->getAttrOfType<IntegerAttr>(kSemaphoreSlotAttrName)) {
        auto cbIndexAttr = op->getAttrOfType<IntegerAttr>(kCBIndexAttrName);
        if (!cbIndexAttr) {
          return op.emitOpError()
                 << "missing static CB index for internal semaphore slot "
                 << slotAttr.getInt();
        }
        cb = tracker->createCBConst(
            loc, rewriter, parentFunc, cbIndexAttr.getInt(),
            "cb_id_internal" + std::to_string(slotAttr.getInt()),
            defaultCBType, std::nullopt, slotAttr.getInt());
      } else {
        return op.emitOpError()
               << "missing semaphore slot metadata for internal semaphore";
      }
    }

    if (!cb)
      return rewriter.notifyMatchFailure(
          op, "missing runtime-arg layout entry for internal loom.semaphore");

    if (expectedCBType && cb.getType() != expectedCBType) {
      return op.emitOpError()
             << "CB type mismatch for loom.semaphore replacement, expected "
             << expectedCBType << " but got " << cb.getType();
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

    // If this source also feeds loom.gather input, gather transport owns
    // consumption and semaphore_give is only a marker.
    if (isSemaphoreGiveForGatherInput(op)) {
      rewriter.eraseOp(op);
      return success();
    }

    // Scalar runtime-site buffers are replaced by per-core runtime scalar
    // arguments. Their semaphore-give is a no-op marker.
    Value sourceMemref = stripMemrefCasts(op.getSource());
    if (auto semTake = sourceMemref.getDefiningOp<::loom::SemaphoreTakeOp>()) {
      if (semTake->hasAttr(kScalarSiteIdAttrName)) {
        rewriter.eraseOp(op);
        return success();
      }
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

    Value semaphoreBase = resolveSemaphoreSourceMemref(op.getSource());
    auto semGiveTiles = getNumTilesForSemaphoreGive(semaphoreBase.getType());
    if (!semGiveTiles)
      semGiveTiles = getNumTilesFromCBType(cbType);
    if (!semGiveTiles) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive tile count for semaphore_give");
    }

    Value numPages = rewriter.create<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), *semGiveTiles);


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

    // Scalar load-sites are replaced by per-core runtime scalar args.
    // No reader-side DRAM->CB movement is emitted for these copies.
    if (op->hasAttr(kScalarSiteIdAttrName)) {
      rewriter.eraseOp(op);
      return success();
    }

    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc)
      return failure();
    auto slot = getCopyBindingSlot(op.getOperation());
    if (!slot)
      return rewriter.notifyMatchFailure(
          op, "missing required copy binding slot on DRAM load");
    CopyBindingData *bindingData =
        tracker->getCopyBindingData(parentFunc.getOperation(), *slot);
    if (!bindingData)
      return rewriter.notifyMatchFailure(
          op, "missing runtime tuple for DRAM load copy binding");

    Value cb = bindingData->cb;
    if (!cb)
      return op.emitOpError("missing CB runtime arg for DRAM load binding slot ")
             << *slot;
    auto cbType = cast<CBType>(cb.getType());
    auto elementType = cbType.getElementType();
    Value numPages;
    std::optional<int64_t> annotatedVecTiles = getAnnotatedVecTiles(op);
    std::optional<int64_t> rank1VecTiles =
        getRank1VecTilesFromMemref(reinterpretCastOp.getResult().getType());

    if (rank1VecTiles) {
      int64_t vecTiles = annotatedVecTiles.value_or(*rank1VecTiles);
      if (vecTiles <= 0) {
        return rewriter.notifyMatchFailure(
            op, "invalid vector tile count for DRAM load");
      }
      numPages =
          rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(),
                                                static_cast<int32_t>(vecTiles));
    } else if (isa<TileType>(elementType)) {
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
      numPages = ceilDivSICompat(rewriter, loc, totalSize, pageSize);
    }

    bindingData->num_tiles = numPages;
    CBReserveBackOp::create(rewriter, loc, cb, bindingData->num_tiles);

    // Determine broadcast direction from the loom.copy broadcast attribute.
    // broadcast : [x, y]
    //   - x > 1, y <= 1 → horizontal broadcast
    //   - x <= 1, y > 1 → vertical broadcast
    //   - x > 1, y > 1  → broadcast all
    //   - otherwise     → unicast
    bool isBroadcast = false;
    auto parsedBroadcast = parseBroadcastAttr(op.getBroadcastAttr());
    if (parsedBroadcast) {
      int64_t xBroadcast = parsedBroadcast->first;
      int64_t yBroadcast = parsedBroadcast->second;

      bool hasHorizontalBroadcast = xBroadcast > 1;
      bool hasVerticalBroadcast = yBroadcast > 1;
      isBroadcast = hasHorizontalBroadcast || hasVerticalBroadcast;
    }
    if (isBroadcast) {
      Operation *parentFunc = op->getParentOfType<func::FuncOp>();
      Value coreX;
      Value coreY;

      coreX = tracker->getCoreCoordForDim(parentFunc, "x");
      coreY = tracker->getCoreCoordForDim(parentFunc, "y");
      if (!coreX || !coreY)
        return op.emitOpError()
               << "missing mapped spatial core coordinates for broadcast; "
                  "expected scf.parallel metadata to define x/y";
  
      coreX = toI32(rewriter, loc, coreX);
      coreY = toI32(rewriter, loc, coreY);

      Value ulX = op.getUlX();
      Value ulY = op.getUlY();
      if (!ulX || !ulY)
        return op.emitOpError()
               << "missing broadcast UL coordinates; expected region "
                  "(UL : [x, y], LR : [...])";
      ulX = toI32(rewriter, loc, ulX);
      ulY = toI32(rewriter, loc, ulY);

      Value isSenderX = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, coreX, ulX);
      Value isSenderY = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, coreY, ulY);
      Value cond = arith::AndIOp::create(rewriter, loc, isSenderX, isSenderY);

      auto ifOp = rewriter.create<scf::IfOp>(loc, cond, true);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(
            ifOp.getThenRegion().front().getTerminator());
        auto [totalSizeBytes, multicast_l1Addr] = dram_read(
            source, loc, rewriter, bindingData);
        if (!totalSizeBytes || !multicast_l1Addr)
          return op.emitOpError("failed to emit DRAM read for multicast send");
        multicast_send(rewriter, loc, bindingData, totalSizeBytes,
                       multicast_l1Addr);

        rewriter.setInsertionPoint(
            ifOp.getElseRegion().front().getTerminator());
        multicast_receive(rewriter, loc, bindingData);
      }
      rewriter.setInsertionPointAfter(ifOp);
    } else {
      auto [totalSizeBytes, multicast_l1Addr] = dram_read(
          source, loc, rewriter, bindingData);
      if (!totalSizeBytes || !multicast_l1Addr)
        return op.emitOpError("failed to emit DRAM read");
    }

    CBPushBackOp::create(rewriter, loc, bindingData->cb,
                         bindingData->num_tiles);
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

    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc)
      return failure();
    auto slot = getCopyBindingSlot(op.getOperation());
    if (!slot)
      return rewriter.notifyMatchFailure(
          op, "missing required copy binding slot on DRAM store");
    CopyBindingData *bindingData =
        tracker->getCopyBindingData(parentFunc.getOperation(), *slot);
    if (!bindingData)
      return rewriter.notifyMatchFailure(
          op, "missing runtime tuple for DRAM store copy binding");

    Value cb = bindingData->cb;
    if (!cb)
      return op.emitOpError("missing CB runtime arg for DRAM store binding slot ")
             << *slot;

    Value baseAddr = bindingData->baseAddr;
    if (!baseAddr)
      return op.emitOpError("missing base address runtime arg for DRAM store");

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
          return op.emitOpError()
                 << "memref.reinterpret_cast for DRAM store must carry "
                    "a dynamic or static offset";
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
    if (isa<TileType>(elementType)) {
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
      numPages = ceilDivSICompat(rewriter, loc, totalSizeBytes, pageSize);
    }

    Value accessorOp = bindingData->tensorAccessor;
    if (!accessorOp)
      return op.emitOpError("missing TensorAccessor runtime arg for DRAM store");

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
      return op.emitOpError()
             << "DRAM store requires explicit strided memref layout; "
                "row-major stride synthesis is not allowed";
    }

    constexpr int64_t kTileDim = 32;
    int64_t numTileRows =
        (shape.size() > 1) ? (shape[shape.size() - 2] + kTileDim - 1) / kTileDim : 1;
    int64_t numTileCols =
        (shape.size() > 0) ? (shape[shape.size() - 1] + kTileDim - 1) / kTileDim : 1;

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
      loc, rewriter.getI32Type(), (strides.size() > 1) ? strides[strides.size() - 2] : 1);
    Value stride1Val = rewriter.create<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), (strides.size() > 0) ? strides[strides.size() - 1] : 1);

    int64_t totalTiles = numTileRows * numTileCols;
    bool enableWriteBarrier = totalTiles > 16;
    int64_t numCores =
        bindingData->parallelCoreCount > 0 ? bindingData->parallelCoreCount : 1;
    int64_t barrierWriteThreshold = 1;
    Value barrierCount =
        rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
    if (enableWriteBarrier) {
      // 2048 represents the tile size.
      barrierWriteThreshold = (512 / numCores) * (1024 + 128) / 2048;
      if (barrierWriteThreshold < 1)
        barrierWriteThreshold = 1;
    }

    scf::ForOp rowLoop =
        scf::ForOp::create(rewriter, loc, loopConst0, numTileRowsVal,
                           loopConst1, ValueRange{l1Addr, barrierCount});
    {
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      Value tileRow = rowLoop.getInductionVar();
      Value crtL1Addr = rowLoop.getRegionIterArgs()[0];
      barrierCount = rowLoop.getRegionIterArgs()[1];

      scf::ForOp colLoop =
          scf::ForOp::create(rewriter, loc, loopConst0, numTileColsVal,
                             loopConst1,
                             ValueRange{crtL1Addr, barrierCount});
      {
        rewriter.setInsertionPointToStart(colLoop.getBody());
        Value tileCol = colLoop.getInductionVar();
        Value innerL1Addr = colLoop.getRegionIterArgs()[0];
        barrierCount = colLoop.getRegionIterArgs()[1];

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
        if (enableWriteBarrier) {
          barrierCount = arith::AddIOp::create(
              rewriter, loc, barrierCount, loopConst1,
              arith::IntegerOverflowFlags::nsw);
          Value barrierThreshold = rewriter.create<arith::ConstantIntOp>(
              loc, rewriter.getI32Type(), barrierWriteThreshold);
          Value shouldWriteBarrier = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq, barrierCount,
              barrierThreshold);

          auto barrierIf = scf::IfOp::create(
              rewriter, loc, TypeRange{rewriter.getI32Type()},
              shouldWriteBarrier, true);
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(
                &barrierIf.getThenRegion().front());
            NocAsyncWriteBarrierOp::create(rewriter, loc);
            scf::YieldOp::create(rewriter, loc, ValueRange{loopConst0});

            rewriter.setInsertionPointToStart(
                &barrierIf.getElseRegion().front());
            scf::YieldOp::create(rewriter, loc, ValueRange{barrierCount});
          }
          rewriter.setInsertionPointAfter(barrierIf);
          barrierCount = barrierIf.getResult(0);
        }
        scf::YieldOp::create(rewriter, loc,
                             ValueRange{nextL1Addr, barrierCount});
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

/**
 * @brief Convert gather transport in writer kernels.
 *
 * @details Writer workers send gathered payload tiles using protocol-specific
 *          semaphore ordering.
 */
struct ConvertLoomGatherTransportOp : public OpConversionPattern<::loom::GatherOp> {
  using OpConversionPattern<::loom::GatherOp>::OpConversionPattern;

  ConvertLoomGatherTransportOp(TypeConverter &typeConverter,
                               MLIRContext *context,
                               std::shared_ptr<CompileArgTracker> tracker,
                               ReduceProtocol protocol)
      : OpConversionPattern<::loom::GatherOp>(typeConverter, context),
        tracker(std::move(tracker)), protocol(protocol) {}

  static LogicalResult validateGatherPlacement(::loom::GatherOp op) {
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (isa<scf::IfOp>(parent)) {
        return op.emitOpError(
            "must be placed outside scf.if so gather transport executes on all "
            "participating cores");
      }
    }
    return success();
  }

  LogicalResult
  matchAndRewrite(::loom::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isWriterKernel(op.getOperation()))
      return failure();
    if (failed(validateGatherPlacement(op)))
      return failure();
    if (adaptor.getOperands().size() < 2)
      return failure();

    // Gather source is payload, gather init is stacked output.
    Value payloadCb = adaptor.getOperands().front();
    Value outCb = adaptor.getOperands()[1];
    if (!isa<CBType>(payloadCb.getType()) || !isa<CBType>(outCb.getType()))
      return failure();

    auto analysis = analyzeReduceTransportRegion(op, rewriter, tracker);
    if (failed(analysis))
      return rewriter.notifyMatchFailure(
          op, "failed to analyze gather transport region");
    auto runtime = materializeReduceTransportRuntime(
        op, payloadCb, outCb, rewriter, tracker, *analysis);
    if (failed(runtime))
      return rewriter.notifyMatchFailure(
          op, "failed to materialize gather transport runtime");

    Location loc = op.getLoc();
    scf::IfOp inRegionIf =
        rewriter.create<scf::IfOp>(loc, analysis->inRegion,
                                   /*withElseRegion=*/false);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&inRegionIf.getThenRegion().front());
      scf::IfOp roleIf = rewriter.create<scf::IfOp>(
          loc, analysis->isReducer, /*withElseRegion=*/true);
      rewriter.setInsertionPointToStart(&roleIf.getThenRegion().front());
      emitReducerReduceTransportSync(rewriter, loc, *analysis, *runtime,
                                     protocol);
      rewriter.setInsertionPointToStart(&roleIf.getElseRegion().front());
      emitWorkerReduceTransport(rewriter, loc, *analysis, *runtime, protocol);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  std::shared_ptr<CompileArgTracker> tracker;
  ReduceProtocol protocol;
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

    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc)
      return failure();
    auto slot = getCopyBindingSlot(op.getOperation());
    if (!slot)
      return rewriter.notifyMatchFailure(
          op, "missing required copy binding slot on compute-side DRAM store");
    CopyBindingData *bindingData =
        tracker->getCopyBindingData(parentFunc.getOperation(), *slot);
    if (!bindingData)
      return rewriter.notifyMatchFailure(
          op, "missing runtime tuple for compute-side DRAM store");

    Value outcb = bindingData->cb;
    if (!outcb || !isa<CBType>(outcb.getType()))
      return op.emitOpError(
          "missing CB runtime arg for compute-side DRAM store target");

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
    MLIRContext *context, std::shared_ptr<CompileArgTracker> tracker,
    ReduceProtocol reduceProtocol) {
  // loom.semaphore / loom.copy patterns.
  patterns.add<ConvertLoomSemaphoreTakeOp>(typeConverter, context, tracker);
  patterns.add<ConvertLoomSemaphoreGiveOp>(typeConverter, context);
  patterns.add<ConvertLoomLoadOp>(typeConverter, context, tracker);
  patterns.add<ConvertLoomStoreOp>(typeConverter, context, tracker);
  patterns.add<ConvertLoomGatherTransportOp>(typeConverter, context,
                                             std::move(tracker), reduceProtocol);
  // Reinterpret cast erasure.
  patterns.add<ConvertReinterpretCastOp>(typeConverter, context);
}

void mlir::loom::populateLoomAllocCleanupPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context) {
  patterns.add<ConvertLoomAllocOp>(typeConverter, context);
}
