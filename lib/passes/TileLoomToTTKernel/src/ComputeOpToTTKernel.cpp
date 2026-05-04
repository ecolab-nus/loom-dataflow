/**
 * @file ComputeOpToTTKernel.cpp
 * @brief Conversion patterns for compute ops to TTKernel.
 */

#include "ComputeOpToTTKernel.h"
#include "FuncOpToTTKernel.h"
#include "TTKernelAttrs.h"
#include "TTKernelUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cstdint>
#include <optional>
#include <tuple>

// Loom dialect headers for Loom ops used in compute lowering.
#define GET_OP_CLASSES
#include "LoomEnums.h.inc"
#include "LoomAttributes.h.inc"
#include "LoomOps.h.inc"

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;
using mlir::loom::CompileArgTracker;

namespace {

enum class FlashAttentionGenericKind {
  Reduction,
  Elementwise
};

/// Elementwise broadcast categories accepted by the generic lowering.
enum class ElementwiseBroadcastKind {
  /// Broadcast over the last tensor dimension.
  RowBroadcast,
  /// Broadcast over the second-last tensor dimension.
  ColBroadcast,
  /// Broadcast over a dimension other than the last two.
  BatchBroadcast
};

/// Metadata describing one elementwise broadcast input.
struct ElementwiseBroadcastInfo {
  /// Input operand index in the generic op.
  unsigned inputIdx = 0;
  /// True when input indexing map is scalar (`()->`), i.e. broadcast all tiles.
  bool isScalar = false;
  /// Output-loop dimension that is broadcast (dropped in input indexing map).
  unsigned droppedDim = 0;
  /// Normalized broadcast category.
  ElementwiseBroadcastKind kind = ElementwiseBroadcastKind::BatchBroadcast;
  /// Tile-count extent for the dropped dimension.
  int64_t droppedDimTiles = 1;
  /// Product of tile-count extents after @p droppedDim.
  int64_t suffixTiles = 1;
};

struct ElementwiseAnalysis {
  Value yieldValue;
  int64_t outTiles = 0;
  SmallVector<std::optional<ElementwiseBroadcastInfo>, 4> broadcastsByInput;
  SmallVector<int64_t, 4> inputWaitTiles;
  llvm::SmallBitVector usedInputs;

  bool needsBinopWithScalar = false;
  bool needsSubBinary = false;
  bool needsAddBinary = false;
  bool needsMulBinary = false;
  bool needsBinaryMax = false;
  bool needsPowBinary = false;
  bool needsRecip = false;
  bool needsLog = false;
  bool needsExp = false;
};

static std::optional<int64_t> getAnnotatedVecTilesFromInput(Value value) {
  Value current = value;
  while (current) {
    if (auto cast = current.getDefiningOp<memref::CastOp>()) {
      current = cast.getSource();
      continue;
    }
    if (auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>()) {
      if (auto tilesAttr = sem->getAttrOfType<IntegerAttr>(kVecTilesAttrName)) {
        int64_t tiles = tilesAttr.getInt();
        if (tiles > 0)
          return tiles;
      }
      current = sem.getSource();
      continue;
    }
    if (auto viewLike = current.getDefiningOp<ViewLikeOpInterface>()) {
      current = viewLike.getViewSource();
      continue;
    }
    break;
  }
  return std::nullopt;
}

static Value stripElementwiseInputWrappers(Value value) {
  Value current = value;
  while (current) {
    if (auto cast = current.getDefiningOp<memref::CastOp>()) {
      current = cast.getSource();
      continue;
    }
    if (auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>()) {
      current = sem.getSource();
      continue;
    }
    if (auto viewLike = current.getDefiningOp<ViewLikeOpInterface>()) {
      current = viewLike.getViewSource();
      continue;
    }
    break;
  }
  return current;
}

static std::optional<int64_t>
getScalarSiteIdForGenericInput(linalg::GenericOp op, unsigned inputIdx) {
  if (inputIdx >= op.getNumDpsInputs())
    return std::nullopt;

  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return std::nullopt;

  Value inputRoot = stripElementwiseInputWrappers(op.getDpsInputs()[inputIdx]);
  if (auto blockArg = dyn_cast<BlockArgument>(inputRoot)) {
    if (blockArg.getOwner() == &parentFunc.front()) {
      if (auto siteAttr = dyn_cast_or_null<IntegerAttr>(parentFunc.getArgAttr(
              blockArg.getArgNumber(), kScalarSiteIdAttrName)))
        return siteAttr.getInt();
    }
  }

  std::optional<int64_t> siteId;
  parentFunc.walk([&](::loom::CopyOp copyOp) {
    if (siteId)
      return;
    auto siteAttr = copyOp->getAttrOfType<IntegerAttr>(kScalarSiteIdAttrName);
    if (!siteAttr)
      return;
    Value destinationRoot = stripElementwiseInputWrappers(copyOp.getDestination());
    if (destinationRoot != inputRoot)
      return;
    siteId = siteAttr.getInt();
  });
  return siteId;
}

static LogicalResult analyzeElementwiseGeneric(linalg::GenericOp op,
                                               ElementwiseAnalysis &analysis);
static bool isSupportedElementwiseGeneric(linalg::GenericOp op) {
  ElementwiseAnalysis analysis;
  return succeeded(analyzeElementwiseGeneric(op, analysis));
}

struct NextUseInBlock {
  Operation *user = nullptr;
  Value usedValue;
};

/**
 * @brief Find the first non-view same-block consumer after `anchorOp`.
 *
 * @details Traverses view-like aliases from `rootValue` and returns the first
 *          non-view user in the same block that appears after `anchorOp`.
 *          Also returns the concrete alias value consumed by that user.
 */
static std::optional<NextUseInBlock>
findNextNonViewUseInSameBlock(Value rootValue, Operation *anchorOp) {
  if (!anchorOp || !anchorOp->getBlock())
    return std::nullopt;

  Block *anchorBlock = anchorOp->getBlock();
  SmallVector<Value, 8> worklist;
  llvm::SmallPtrSet<Value, 16> seenValues;
  std::optional<NextUseInBlock> nextUse;

  worklist.push_back(rootValue);
  seenValues.insert(rootValue);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (Operation *user : current.getUsers()) {
      if (user == anchorOp || user->getBlock() != anchorBlock ||
          !anchorOp->isBeforeInBlock(user))
        continue;

      if (isa<ViewLikeOpInterface>(user)) {
        for (Value result : user->getResults())
          if (seenValues.insert(result).second)
            worklist.push_back(result);
        continue;
      }

      if (!nextUse || user->isBeforeInBlock(nextUse->user))
        nextUse = NextUseInBlock{user, current};
    }
  }

  return nextUse;
}

template <typename OpTy>
static bool bodyHasOp(linalg::GenericOp op) {
  for (Operation &inner : op.getRegion().front().without_terminator())
    if (isa<OpTy>(inner))
      return true;
  return false;
}

static std::optional<FlashAttentionGenericKind>
classifyFlashAttentionGeneric(linalg::GenericOp op) {
  if (op.getNumDpsInits() != 1)
    return std::nullopt;

  if (llvm::any_of(op.getIteratorTypesArray(), [](utils::IteratorType type) {
        return type == utils::IteratorType::reduction;
      }))
    return FlashAttentionGenericKind::Reduction;

  if (isSupportedElementwiseGeneric(op))
    return FlashAttentionGenericKind::Elementwise;

  return std::nullopt;
}

static LogicalResult rewriteReduceGeneric(linalg::GenericOp op,
                                          linalg::GenericOp::Adaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          llvm::DenseMap<Value, int64_t> &waitState);

enum class TileReductionCombineOp {
  Sum,
  Max
};

struct TileReductionAnalysis {
  TileReductionCombineOp combineOp = TileReductionCombineOp::Sum;
};

/**
 * @brief Distinguishes tile-generic body block arguments by semantic role.
 */
enum class TileGenericBodyOperand {
  Input,
  Accumulator
};

static LogicalResult analyzeTileReductionGeneric(
    linalg::GenericOp op, TileReductionAnalysis &analysis);

/**
 * @brief Check for the [reduction, parallel, parallel] iterator pattern.
 */
static bool isTileGenericOp(linalg::GenericOp op) {
  auto iteratorTypes = op.getIteratorTypesArray();
  if (iteratorTypes.size() < 3)
    return false;
  if (iteratorTypes[0] != utils::IteratorType::reduction)
    return false;
  for (utils::IteratorType type : llvm::drop_begin(iteratorTypes))
    if (type != utils::IteratorType::parallel)
      return false;
  return true;
}

/**
 * @brief Entry-point for [reduction, parallel, parallel] generic lowering.
 *
 * @details Keeps the partial accumulator in DST registers for the complete
 *          reduction loop. Only the final accumulated tile is packed to the
 *          output CB, avoiding repeated BF16 round-trips through CB storage.
 */
static LogicalResult tileGenericOp(
    linalg::GenericOp op, linalg::GenericOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter,
    llvm::DenseMap<Value, int64_t> &waitState) {
  if (adaptor.getInputs().size() != 1 || adaptor.getOutputs().size() != 1)
    return failure();

  Value inCb = adaptor.getInputs().front();
  Value outCb = adaptor.getOutputs().front();
  if (!isa<CBType>(inCb.getType()) || !isa<CBType>(outCb.getType()))
    return op.emitOpError()
           << "tile generic lowering requires CB-typed input and output";

  TileReductionAnalysis analysis;
  if (failed(analyzeTileReductionGeneric(op, analysis)))
    return op.emitOpError() << "unsupported tile reduction body";

  auto inType = dyn_cast<ShapedType>(op.getDpsInputs()[0].getType());
  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  if (!inType || !outType || !inType.hasStaticShape() || !outType.hasStaticShape())
    return failure();

  ArrayRef<int64_t> inShape = inType.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();
  if (inShape.size() != outShape.size() + 1 || inShape[0] <= 0)
    return failure();
  for (size_t dim = 0; dim < outShape.size(); ++dim) {
    if (inShape[dim + 1] != outShape[dim])
      return failure();
  }

  int64_t outTiles = 1;
  unsigned outRank = outShape.size();
  unsigned tiledStartDim = outRank > 2 ? outRank - 2 : 0;
  for (unsigned dim = 0; dim < outRank; ++dim) {
    int64_t dimSize = outShape[dim];
    if (dimSize <= 0)
      return failure();

    if (dim >= tiledStartDim) {
      auto dimTiles = ceilDiv32(dimSize);
      if (!dimTiles || *dimTiles <= 0)
        return failure();
      outTiles *= *dimTiles;
    } else {
      outTiles *= dimSize;
    }
  }
  int64_t reduceSlices = inShape[0];
  int64_t inputTiles = reduceSlices * outTiles;

  Location loc = op.getLoc();
  auto i32 = [&](int64_t value) -> Value {
    return rewriter.create<arith::ConstantIntOp>(loc, value, 32);
  };
  Value zeroI32 = i32(0);
  Value oneI32 = i32(1);
  Value outTilesV = i32(outTiles);
  Value reduceSlicesV = i32(reduceSlices);

  auto emitWaitFront = [&](Value cb, int64_t tiles) {
    if (tiles <= 0)
      return;
    int64_t outstanding = 0;
    auto it = waitState.find(cb);
    if (it != waitState.end())
      outstanding = it->second;
    if (outstanding >= tiles)
      return;
    CBWaitFrontOp::create(rewriter, loc, cb, i32(tiles));
    waitState[cb] = tiles;
  };

  emitWaitFront(inCb, inputTiles);

  CBReserveBackOp::create(rewriter, loc, outCb, outTilesV);
  rewriter.create<CopyTileInitOp>(loc, inCb);
  if (analysis.combineOp == TileReductionCombineOp::Sum)
    rewriter.create<AddBinaryTilesInitOp>(loc);
  else
    rewriter.create<BinaryMaxTileInitOp>(loc);

  scf::ForOp outTileLoop =
      scf::ForOp::create(rewriter, loc, zeroI32, outTilesV, oneI32);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(outTileLoop.getBody());
    Value outTileIdx = outTileLoop.getInductionVar();

    TileRegsAcquireOp::create(rewriter, loc);
    rewriter.create<CopyTileOp>(loc, inCb, outTileIdx, zeroI32);

    scf::ForOp reduceLoop =
        scf::ForOp::create(rewriter, loc, oneI32, reduceSlicesV, oneI32);
    {
      OpBuilder::InsertionGuard reduceGuard(rewriter);
      rewriter.setInsertionPointToStart(reduceLoop.getBody());
      Value reduceIdx = reduceLoop.getInductionVar();
      Value sliceOffset =
          arith::MulIOp::create(rewriter, loc, reduceIdx, outTilesV);
      Value inTileIdx =
          arith::AddIOp::create(rewriter, loc, sliceOffset, outTileIdx);
      rewriter.create<CopyTileOp>(loc, inCb, inTileIdx, oneI32);
      if (analysis.combineOp == TileReductionCombineOp::Sum) {
        rewriter.create<AddBinaryTilesOp>(loc, zeroI32, oneI32, zeroI32);
      } else {
        rewriter.create<BinaryMaxTileOp>(loc, zeroI32, oneI32, zeroI32);
      }
    }
    rewriter.setInsertionPointAfter(reduceLoop);
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);
    PackTileOp::create(rewriter, loc, zeroI32, outCb, outTileIdx);
    TileRegsReleaseOp::create(rewriter, loc);
  }
  CBPushBackOp::create(rewriter, loc, outCb, outTilesV);
  waitState[outCb] = 0;
  //CBPopFrontOp::create(rewriter, loc, inCb, i32(inputTiles));
  waitState[inCb] = 0;

  rewriter.eraseOp(op);
  return success();
}

static bool isIdentityMapForRank(AffineMap map, unsigned rank) {
  if (!map || map.getNumDims() != rank || map.getNumSymbols() != 0 ||
      map.getNumResults() != rank)
    return false;
  for (unsigned i = 0; i < rank; ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i));
    if (!dimExpr || dimExpr.getPosition() != i)
      return false;
  }
  return true;
}

static bool hasAllIdentityMapsForRank(linalg::GenericOp op, unsigned rank) {
  if (op.getNumLoops() != rank)
    return false;
  for (AffineMap map : op.getIndexingMapsArray())
    if (!isIdentityMapForRank(map, rank))
      return false;
  return true;
}

static bool isScalarInputMapForRank(AffineMap map, unsigned rank) {
  return map && map.getNumDims() == rank && map.getNumSymbols() == 0 &&
         map.getNumResults() == 0;
}

/**
 * @brief Return the dropped output dimension for a pure broadcast map.
 *
 * @details A supported broadcast map is rank-1 and consists only of strictly
 *          increasing affine dim expressions, i.e. identity with exactly one
 *          missing dimension.
 */
static std::optional<unsigned> getDroppedDimFromBroadcastMap(AffineMap map,
                                                             unsigned rank) {
  if (!map || map.getNumDims() != rank || map.getNumSymbols() != 0 ||
      map.getNumResults() != rank - 1 || rank == 0)
    return std::nullopt;

  llvm::SmallBitVector seenDims(rank);
  unsigned prevPos = 0;
  bool hasPrev = false;
  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return std::nullopt;
    unsigned pos = dimExpr.getPosition();
    if (pos >= rank || seenDims.test(pos))
      return std::nullopt;
    if (hasPrev && pos <= prevPos)
      return std::nullopt;
    seenDims.set(pos);
    prevPos = pos;
    hasPrev = true;
  }

  for (unsigned dim = 0; dim < rank; ++dim) {
    if (!seenDims.test(dim))
      return dim;
  }
  return std::nullopt;
}

/// Classify broadcast kind by dropped output dimension.
static ElementwiseBroadcastKind
classifyElementwiseBroadcastKind(unsigned droppedDim, unsigned rank) {
  if (rank >= 1 && droppedDim == rank - 1)
    return ElementwiseBroadcastKind::RowBroadcast;
  if (rank >= 2 && droppedDim == rank - 2)
    return ElementwiseBroadcastKind::ColBroadcast;
  return ElementwiseBroadcastKind::BatchBroadcast;
}

/**
 * @brief Map analysis broadcast kind to TTKernel unary-broadcast enum.
 *
 * @details TTKernel naming follows hardware lanes (`col`/`row`), so the
 *          semantic mapping from dropped dimensions is:
 *          - RowBroadcast (drop last dim)      -> `BcastType::Col`
 *          - ColBroadcast (drop second-last)   -> `BcastType::Row`
 *          - BatchBroadcast                     -> no unary broadcast op
 */
static std::optional<BcastType>
getUnaryBcastType(ElementwiseBroadcastKind kind) {
  switch (kind) {
  case ElementwiseBroadcastKind::RowBroadcast:
    return BcastType::Col;
  case ElementwiseBroadcastKind::ColBroadcast:
    return BcastType::Row;
  case ElementwiseBroadcastKind::BatchBroadcast:
    return std::nullopt;
  }
  return std::nullopt;
}

/// Product of tile extents strictly after @p droppedDim.
static std::optional<int64_t>
getSuffixTilesAfterDim(ArrayRef<int64_t> outTileShape, unsigned droppedDim) {
  if (droppedDim >= outTileShape.size())
    return std::nullopt;
  int64_t suffix = 1;
  for (unsigned dim = droppedDim + 1; dim < outTileShape.size(); ++dim) {
    if (outTileShape[dim] <= 0)
      return std::nullopt;
    suffix *= outTileShape[dim];
  }
  return suffix;
}

static std::optional<unsigned> getBodyInputIndex(linalg::GenericOp op, Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != &op.getRegion().front())
    return std::nullopt;
  if (blockArg.getArgNumber() >= op.getNumDpsInputs())
    return std::nullopt;
  return blockArg.getArgNumber();
}

static std::optional<unsigned> getProducerBroadcastDim(Value value) {
  Value current = value;
  llvm::SmallPtrSet<Value, 8> visited;
  while (current && visited.insert(current).second) {
    if (auto broadcastOp = current.getDefiningOp<::loom::BroadcastOp>())
      return static_cast<unsigned>(broadcastOp.getDim());

    if (auto castOp = current.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (auto dimAttr =
              castOp->getAttrOfType<IntegerAttr>(kBroadcastDimAttrName)) {
        int64_t dim = dimAttr.getInt();
        if (dim < 0)
          return std::nullopt;
        return static_cast<unsigned>(dim);
      }
      if (castOp.getNumOperands() == 1) {
        current = castOp.getOperand(0);
        continue;
      }
      break;
    }

    if (auto cast = current.getDefiningOp<memref::CastOp>()) {
      current = cast.getSource();
      continue;
    }
    if (auto sem = current.getDefiningOp<::loom::SemaphoreTakeOp>()) {
      current = sem.getSource();
      continue;
    }
    if (auto viewLike = current.getDefiningOp<ViewLikeOpInterface>()) {
      current = viewLike.getViewSource();
      continue;
    }
    break;
  }
  return std::nullopt;
}

/// Unwrap temporary CB-type bridges created for loom.broadcast logical views.
static Value stripBroadcastBridgeCast(Value value) {
  auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getNumOperands() != 1)
    return value;
  if (!cast->hasAttr(kBroadcastDimAttrName))
    return value;
  Value source = cast.getOperand(0);
  if (!isa<CBType>(source.getType()))
    return value;
  return source;
}

static std::optional<float> getConstFloatValue(Value value) {
  if (auto cst = value.getDefiningOp<arith::ConstantFloatOp>()) {
    auto attr = dyn_cast<FloatAttr>(cst.getValue());
    if (!attr)
      return std::nullopt;
    return static_cast<float>(attr.getValueAsDouble());
  }

  // Accept casted float constants (e.g. f32 const -> truncf -> f16 scalar).
  if (auto trunc = value.getDefiningOp<arith::TruncFOp>())
    return getConstFloatValue(trunc.getIn());
  if (auto ext = value.getDefiningOp<arith::ExtFOp>())
    return getConstFloatValue(ext.getIn());

  return std::nullopt;
}

/// Tile-shape metadata for elementwise generics where only innermost two dims
/// are tiled (32x32), and outer dims are already tile-count dimensions.
struct ElementwiseTileShapeInfo {
  SmallVector<int64_t, 4> dimExtents;
  int64_t totalTiles = 0;
};

static std::optional<ElementwiseTileShapeInfo>
getElementwiseTileShapeInfo(ShapedType type) {
  if (!type || !type.hasStaticShape())
    return std::nullopt;

  unsigned rank = type.getRank();
  unsigned tiledStartDim = rank > 2 ? rank - 2 : 0;

  ElementwiseTileShapeInfo info;
  info.totalTiles = 1;
  info.dimExtents.reserve(rank);
  for (unsigned dim = 0; dim < rank; ++dim) {
    int64_t dimSize = type.getShape()[dim];
    if (dimSize <= 0)
      return std::nullopt;

    int64_t dimExtent = dimSize;
    if (dim >= tiledStartDim) {
      auto dimTiles = ceilDiv32(dimSize);
      if (!dimTiles || *dimTiles <= 0)
        return std::nullopt;
      dimExtent = *dimTiles;
    }

    info.dimExtents.push_back(dimExtent);
    info.totalTiles *= dimExtent;
  }

  return info;
}

static std::optional<int64_t> getTileDim(ShapedType type, unsigned dim) {
  if (!type.hasStaticShape() || dim >= type.getRank())
    return std::nullopt;
  return ceilDiv32(type.getShape()[dim]);
}

struct MatmulTileInfo {
  int64_t batchSize = 1;
  int64_t rt = 0;
  int64_t kt = 0;
  int64_t ct = 0;
  int64_t nt = 0;
  int64_t in0TilesPerBatch = 0;
  int64_t in1TilesPerBatch = 0;
  int64_t outTilesPerBatch = 0;
  int64_t in0TilesTotal = 0;
  int64_t in1TilesTotal = 0;
  int64_t outTilesTotal = 0;
};

/**
 * @brief Analyze rank-3 batch_matmul shapes into tile metadata.
 *
 * @details `linalg.matmul` is normalized to batch-1 `linalg.batch_matmul`
 *          earlier in the pipeline, so this helper only accepts rank-3 shapes.
 */
static std::optional<MatmulTileInfo>
getMatmulTileInfo(ShapedType lhsType, ShapedType rhsType, ShapedType outType) {
  if (!lhsType || !rhsType || !outType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || !outType.hasStaticShape())
    return std::nullopt;

  MatmulTileInfo info;

  auto toI64 = [](std::optional<int64_t> value) -> std::optional<int64_t> {
    if (!value || *value <= 0)
      return std::nullopt;
    return value;
  };

  if (lhsType.getRank() == 3 && rhsType.getRank() == 3 &&
      outType.getRank() == 3) {
    ArrayRef<int64_t> lhs = lhsType.getShape();
    ArrayRef<int64_t> rhs = rhsType.getShape();
    ArrayRef<int64_t> out = outType.getShape();
    if (lhs[0] != rhs[0] || lhs[0] != out[0] || lhs[1] != out[1] ||
        lhs[2] != rhs[1] || rhs[2] != out[2] || lhs[0] <= 0)
      return std::nullopt;

    auto rt = toI64(getTileDim(lhsType, 1));
    auto kt = toI64(getTileDim(lhsType, 2));
    auto ct = toI64(getTileDim(rhsType, 2));
    auto nt = toI64(getTileDim(outType, 2));
    if (!rt || !kt || !ct || !nt)
      return std::nullopt;

    info.batchSize = lhs[0];
    info.rt = *rt;
    info.kt = *kt;
    info.ct = *ct;
    info.nt = *nt;
  } else {
    return std::nullopt;
  }

  info.in0TilesPerBatch = info.rt * info.kt;
  info.in1TilesPerBatch = info.kt * info.ct;
  info.outTilesPerBatch = info.rt * info.nt;
  info.in0TilesTotal = info.in0TilesPerBatch * info.batchSize;
  info.in1TilesTotal = info.in1TilesPerBatch * info.batchSize;
  info.outTilesTotal = info.outTilesPerBatch * info.batchSize;
  if (info.in0TilesPerBatch <= 0 || info.in1TilesPerBatch <= 0 ||
      info.outTilesPerBatch <= 0 || info.in0TilesTotal <= 0 ||
      info.in1TilesTotal <= 0 || info.outTilesTotal <= 0)
    return std::nullopt;

  return info;
}

static Value getScalarBitsFromFloat(ConversionPatternRewriter &rewriter,
                                    Location loc, float value) {
  Value f32Const = rewriter.create<arith::ConstantFloatOp>(
      loc, rewriter.getF32Type(), llvm::APFloat(value));
  return rewriter.create<arith::BitcastOp>(loc, rewriter.getI32Type(), f32Const);
}

static bool hasOutputAlias(Value outCb, ValueRange inCbs) {
  return llvm::is_contained(inCbs, outCb);
}

static int64_t getOutstandingWaitTiles(const llvm::DenseMap<Value, int64_t> &state,
                                       Value cb) {
  auto it = state.find(cb);
  return it == state.end() ? 0 : it->second;
}

static void emitWaitFrontIfNeeded(ConversionPatternRewriter &rewriter, Location loc,
                                  Value cb, int64_t tiles,
                                  llvm::DenseMap<Value, int64_t> &state) {
  if (tiles <= 0)
    return;
  int64_t outstanding = getOutstandingWaitTiles(state, cb);
  if (outstanding >= tiles)
    return;
  CBWaitFrontOp::create(rewriter, loc, cb, i32Const(rewriter, loc, tiles));
  state[cb] = tiles;
}

static void copyTile(ConversionPatternRewriter &rewriter, Location loc,
                     Value inputCb, Value outputCb, Value tiles,
                     bool popInputCb = true) {
  Value zero = i32Const(rewriter, loc, 0);
  Value one = i32Const(rewriter, loc, 1);
  CBWaitFrontOp::create(rewriter, loc, inputCb, tiles);
  CBReserveBackOp::create(rewriter, loc, outputCb, tiles);
  rewriter.create<CopyTileInitOp>(loc, inputCb);
  scf::ForOp tileLoop =
      scf::ForOp::create(rewriter, loc, zero, tiles, one);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(tileLoop.getBody());
    Value tileIdx = tileLoop.getInductionVar();
    TileRegsAcquireOp::create(rewriter, loc);
    rewriter.create<CopyTileOp>(loc, inputCb, tileIdx, zero);
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);
    PackTileOp::create(rewriter, loc, zero, outputCb, tileIdx);
    TileRegsReleaseOp::create(rewriter, loc);
  }
  if (popInputCb) {
    // Default behavior consumes input tiles once the copy is complete.
    CBPopFrontOp::create(rewriter, loc, inputCb, tiles);
  }
}

template <typename BuilderFn>
static LogicalResult emitElementwiseTiles(
    ConversionPatternRewriter &rewriter, Location loc, Value outCb, int64_t outTiles,
    BuilderFn &&builder) {
  Value zeroI32 = i32Const(rewriter, loc, 0);
  Value oneI32 = i32Const(rewriter, loc, 1);
  Value outTilesV = i32Const(rewriter, loc, outTiles);
  scf::ForOp tileLoop =
      scf::ForOp::create(rewriter, loc, zeroI32, outTilesV, oneI32);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(tileLoop.getBody());
    Value tileIdx = tileLoop.getInductionVar();

    TileRegsAcquireOp::create(rewriter, loc);
    if (failed(builder(tileIdx)))
      return failure();
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);
    PackTileOp::create(rewriter, loc, zeroI32, outCb, tileIdx);
    TileRegsReleaseOp::create(rewriter, loc);
  }
  return success();
}

static bool isGreaterCmp(arith::CmpFPredicate pred) {
  return pred == arith::CmpFPredicate::OGT ||
         pred == arith::CmpFPredicate::OGE ||
         pred == arith::CmpFPredicate::UGT ||
         pred == arith::CmpFPredicate::UGE;
}

static bool isLessCmp(arith::CmpFPredicate pred) {
  return pred == arith::CmpFPredicate::OLT ||
         pred == arith::CmpFPredicate::OLE ||
         pred == arith::CmpFPredicate::ULT ||
         pred == arith::CmpFPredicate::ULE;
}

static bool matchMaxSelect(arith::SelectOp selectOp, Value &lhs, Value &rhs) {
  auto cmp = selectOp.getCondition().getDefiningOp<arith::CmpFOp>();
  if (!cmp)
    return false;

  arith::CmpFPredicate pred = cmp.getPredicate();
  Value cmpLhs = cmp.getLhs();
  Value cmpRhs = cmp.getRhs();
  Value trueV = selectOp.getTrueValue();
  Value falseV = selectOp.getFalseValue();

  if (isGreaterCmp(pred) && trueV == cmpLhs && falseV == cmpRhs) {
    lhs = cmpLhs;
    rhs = cmpRhs;
    return true;
  }

  if (isLessCmp(pred) && trueV == cmpRhs && falseV == cmpLhs) {
    lhs = cmpLhs;
    rhs = cmpRhs;
    return true;
  }

  return false;
}

enum class GenericExprFeature {
  BinopWithScalar,
  SubBinary,
  AddBinary,
  MulBinary,
  BinaryMax,
  PowBinary,
  Recip,
  Log,
  Exp,
  Fill
};

/**
 * @brief Recursively analyze a generic expression tree.
 *
 * @details This helper is shared by FlashAttention elementwise analysis and
 *          tile-generic analysis. Callers provide leaf handling and feature
 *          flag plumbing while this function walks the op tree.
 */
template <typename LeafHandlerT, typename ConstHandlerT,
          typename PowBaseConstHandlerT, typename FlagHandlerT>
static LogicalResult analyzeGenericExprTree(
    Value exprValue, linalg::GenericOp op, LeafHandlerT &&handleLeaf,
    ConstHandlerT &&handleConst, PowBaseConstHandlerT &&handlePowBaseConst,
    FlagHandlerT &&setFlag, bool allowMaximumFOp) {
  auto recurse = [&](auto &&self, Value value) -> LogicalResult {
    if (handleLeaf(value))
      return success();

    if (getConstFloatValue(value).has_value()) {
      handleConst();
      return success();
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp || defOp->getBlock() != &op.getRegion().front())
      return failure();

    if (auto mulOp = dyn_cast<arith::MulFOp>(defOp)) {
      auto lhsConst = getConstFloatValue(mulOp.getLhs());
      auto rhsConst = getConstFloatValue(mulOp.getRhs());
      if (lhsConst && !rhsConst) {
        setFlag(GenericExprFeature::BinopWithScalar);
        return self(self, mulOp.getRhs());
      }
      if (rhsConst && !lhsConst) {
        setFlag(GenericExprFeature::BinopWithScalar);
        return self(self, mulOp.getLhs());
      }
      setFlag(GenericExprFeature::MulBinary);
      if (failed(self(self, mulOp.getLhs())))
        return failure();
      return self(self, mulOp.getRhs());
    }

    if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
      setFlag(GenericExprFeature::AddBinary);
      if (failed(self(self, addOp.getLhs())))
        return failure();
      return self(self, addOp.getRhs());
    }

    if (auto subOp = dyn_cast<arith::SubFOp>(defOp)) {
      setFlag(GenericExprFeature::SubBinary);
      if (failed(self(self, subOp.getLhs())))
        return failure();
      return self(self, subOp.getRhs());
    }

    if (auto divOp = dyn_cast<arith::DivFOp>(defOp)) {
      setFlag(GenericExprFeature::Recip);
      setFlag(GenericExprFeature::MulBinary);
      if (failed(self(self, divOp.getLhs())))
        return failure();
      return self(self, divOp.getRhs());
    }

    if (auto powOp = dyn_cast<math::PowFOp>(defOp)) {
      if (!getConstFloatValue(powOp.getLhs()).has_value())
        return failure();
      handlePowBaseConst();
      setFlag(GenericExprFeature::PowBinary);
      return self(self, powOp.getRhs());
    }

    if (auto logOp = dyn_cast<math::LogOp>(defOp)) {
      setFlag(GenericExprFeature::Log);
      return self(self, logOp.getOperand());
    }

    if (auto expOp = dyn_cast<math::ExpOp>(defOp)) {
      setFlag(GenericExprFeature::Exp);
      return self(self, expOp.getOperand());
    }

    if (auto maxOp = dyn_cast<arith::MaximumFOp>(defOp)) {
      if (!allowMaximumFOp)
        return failure();
      setFlag(GenericExprFeature::BinaryMax);
      if (failed(self(self, maxOp.getLhs())))
        return failure();
      return self(self, maxOp.getRhs());
    }

    if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
      Value lhs;
      Value rhs;
      if (!matchMaxSelect(selectOp, lhs, rhs))
        return failure();
      setFlag(GenericExprFeature::BinaryMax);
      if (failed(self(self, lhs)))
        return failure();
      return self(self, rhs);
    }

    return failure();
  };

  return recurse(recurse, exprValue);
}

/**
 * @brief Resolve tile-generic body values to input or accumulator operands.
 */
static std::optional<TileGenericBodyOperand>
getTileGenericBodyOperand(linalg::GenericOp op, Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != &op.getRegion().front())
    return std::nullopt;

  unsigned argNumber = blockArg.getArgNumber();
  if (argNumber == 0)
    return TileGenericBodyOperand::Input;

  if (argNumber == op.getNumDpsInputs())
    return TileGenericBodyOperand::Accumulator;

  return std::nullopt;
}

static bool hasOnlyBodyOps(linalg::GenericOp op, Operation *first,
                           Operation *second = nullptr) {
  for (Operation &inner : op.getRegion().front().without_terminator())
    if (&inner != first && &inner != second)
      return false;
  return true;
}

static bool matchTileReductionOperands(linalg::GenericOp op, Value lhs,
                                       Value rhs) {
  auto lhsOperand = getTileGenericBodyOperand(op, lhs);
  auto rhsOperand = getTileGenericBodyOperand(op, rhs);
  if (!lhsOperand || !rhsOperand)
    return false;

  return (*lhsOperand == TileGenericBodyOperand::Input &&
          *rhsOperand == TileGenericBodyOperand::Accumulator) ||
         (*lhsOperand == TileGenericBodyOperand::Accumulator &&
          *rhsOperand == TileGenericBodyOperand::Input);
}

static LogicalResult analyzeTileReductionGeneric(
    linalg::GenericOp op, TileReductionAnalysis &analysis) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
    return failure();

  auto yieldOp = dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return failure();

  Value yielded = yieldOp.getValues().front();
  Operation *defOp = yielded.getDefiningOp();
  if (!defOp || defOp->getBlock() != &op.getRegion().front())
    return failure();

  if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
    if (!hasOnlyBodyOps(op, addOp.getOperation()) ||
        !matchTileReductionOperands(op, addOp.getLhs(), addOp.getRhs()))
      return failure();
    analysis.combineOp = TileReductionCombineOp::Sum;
    return success();
  }

  if (auto maxOp = dyn_cast<arith::MaximumFOp>(defOp)) {
    if (!hasOnlyBodyOps(op, maxOp.getOperation()) ||
        !matchTileReductionOperands(op, maxOp.getLhs(), maxOp.getRhs()))
      return failure();
    analysis.combineOp = TileReductionCombineOp::Max;
    return success();
  }

  if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
    Value lhs;
    Value rhs;
    if (!matchMaxSelect(selectOp, lhs, rhs) ||
        !matchTileReductionOperands(op, lhs, rhs))
      return failure();
    Operation *cmpOp = selectOp.getCondition().getDefiningOp();
    if (!cmpOp || cmpOp->getBlock() != &op.getRegion().front() ||
        !hasOnlyBodyOps(op, selectOp.getOperation(), cmpOp))
      return failure();
    analysis.combineOp = TileReductionCombineOp::Max;
    return success();
  }

  return failure();
}

/**
 * @brief Shared recursive emission for generic expression trees.
 *
 * @details This helper emits all non-leaf expression operators shared by
 *          FlashAttention elementwise and tile-generic lowering. Callers
 *          provide leaf materialization and operand-order policy.
 */
template <typename LeafEmitterT, typename RhsFirstPolicyT,
          typename RuntimeScalarLookupT>
static LogicalResult emitGenericExprToRegImpl(
    Value exprValue, int dstReg, int tmpRegA, int tmpRegB, linalg::GenericOp op,
    linalg::GenericOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
    Location loc, bool emitInlineInitOps, bool allowMaximumFOp,
    LeafEmitterT &&emitLeafToReg, RhsFirstPolicyT &&emitRhsFirstForOperands,
    RuntimeScalarLookupT &&lookupRuntimeScalarBits) {
  auto recurse =
      [&](auto &&self, Value value, int curDstReg, int curTmpRegA,
          int curTmpRegB) -> LogicalResult {
    Value curDstRegVal = i32Const(rewriter, loc, curDstReg);
    Value curTmpRegAVal = i32Const(rewriter, loc, curTmpRegA);

    bool handledLeaf = false;
    if (failed(emitLeafToReg(value, curDstReg, handledLeaf)))
      return failure();
    if (handledLeaf)
      return success();

    if (auto constFloat = getConstFloatValue(value)) {
      if (emitInlineInitOps)
        rewriter.create<FillTileInitOp>(loc);
      Value constVal = rewriter.create<arith::ConstantFloatOp>(
          loc, rewriter.getF32Type(), llvm::APFloat(*constFloat));
      rewriter.create<FillTileOp>(loc, curDstRegVal, constVal);
      return success();
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp || defOp->getBlock() != &op.getRegion().front())
      return failure();

    if (auto mulOp = dyn_cast<arith::MulFOp>(defOp)) {
      auto lhsConst = getConstFloatValue(mulOp.getLhs());
      auto rhsConst = getConstFloatValue(mulOp.getRhs());
      if (lhsConst && !rhsConst) {
        if (failed(self(self, mulOp.getRhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (emitInlineInitOps)
          rewriter.create<BinopWithScalarTileInitOp>(loc);
        rewriter.create<MulUnaryTileOp>(
            loc, curDstRegVal, getScalarBitsFromFloat(rewriter, loc, *lhsConst));
        return success();
      }
      if (rhsConst && !lhsConst) {
        if (failed(self(self, mulOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (emitInlineInitOps)
          rewriter.create<BinopWithScalarTileInitOp>(loc);
        rewriter.create<MulUnaryTileOp>(
            loc, curDstRegVal, getScalarBitsFromFloat(rewriter, loc, *rhsConst));
        return success();
      }
      Value lhsRuntimeScalar = lookupRuntimeScalarBits(mulOp.getLhs());
      Value rhsRuntimeScalar = lookupRuntimeScalarBits(mulOp.getRhs());
      if (lhsRuntimeScalar && !rhsRuntimeScalar) {
        if (failed(self(self, mulOp.getRhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (emitInlineInitOps)
          rewriter.create<BinopWithScalarTileInitOp>(loc);
        rewriter.create<MulUnaryTileOp>(loc, curDstRegVal, lhsRuntimeScalar);
        return success();
      }
      if (rhsRuntimeScalar && !lhsRuntimeScalar) {
        if (failed(self(self, mulOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (emitInlineInitOps)
          rewriter.create<BinopWithScalarTileInitOp>(loc);
        rewriter.create<MulUnaryTileOp>(loc, curDstRegVal, rhsRuntimeScalar);
        return success();
      }
      if (lhsRuntimeScalar && rhsRuntimeScalar)
        return failure();

      bool emitRhsFirst = emitRhsFirstForOperands(mulOp.getLhs(), mulOp.getRhs());
      if (emitRhsFirst) {
        if (failed(self(self, mulOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
        if (failed(self(self, mulOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
      } else {
        if (failed(self(self, mulOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (failed(self(self, mulOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
      }
      if (emitInlineInitOps)
        rewriter.create<MulBinaryTilesInitOp>(loc);
      rewriter.create<MulBinaryTilesOp>(loc, curDstRegVal, curTmpRegAVal,
                                        curDstRegVal);
      return success();
    }

    if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
      bool emitRhsFirst = emitRhsFirstForOperands(addOp.getLhs(), addOp.getRhs());
      if (emitRhsFirst) {
        if (failed(self(self, addOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
        if (failed(self(self, addOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
      } else {
        if (failed(self(self, addOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (failed(self(self, addOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
      }
      if (emitInlineInitOps)
        rewriter.create<AddBinaryTilesInitOp>(loc);
      rewriter.create<AddBinaryTilesOp>(loc, curDstRegVal, curTmpRegAVal,
                                        curDstRegVal);
      return success();
    }

    if (auto subOp = dyn_cast<arith::SubFOp>(defOp)) {
      bool emitRhsFirst = emitRhsFirstForOperands(subOp.getLhs(), subOp.getRhs());
      if (emitRhsFirst) {
        if (failed(self(self, subOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
        if (failed(self(self, subOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
      } else {
        if (failed(self(self, subOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (failed(self(self, subOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
      }
      if (emitInlineInitOps)
        rewriter.create<SubBinaryTilesInitOp>(loc);
      rewriter.create<SubBinaryTilesOp>(loc, curDstRegVal, curTmpRegAVal,
                                        curDstRegVal);
      return success();
    }

    if (auto divOp = dyn_cast<arith::DivFOp>(defOp)) {
      bool emitRhsFirst = emitRhsFirstForOperands(divOp.getLhs(), divOp.getRhs());
      if (emitRhsFirst) {
        if (failed(self(self, divOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
        if (failed(self(self, divOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
      } else {
        if (failed(self(self, divOp.getLhs(), curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (failed(self(self, divOp.getRhs(), curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
      }
      if (emitInlineInitOps)
        rewriter.create<RecipTileInitOp>(loc);
      rewriter.create<RecipTileOp>(loc, curTmpRegAVal);
      if (emitInlineInitOps)
        rewriter.create<MulBinaryTilesInitOp>(loc);
      rewriter.create<MulBinaryTilesOp>(loc, curDstRegVal, curTmpRegAVal,
                                        curDstRegVal);
      return success();
    }

    if (auto powOp = dyn_cast<math::PowFOp>(defOp)) {
      auto lhsConst = getConstFloatValue(powOp.getLhs());
      if (!lhsConst)
        return failure();
      if (failed(self(self, powOp.getRhs(), curDstReg, curTmpRegA, curTmpRegB)))
        return failure();
      Value lhsConstVal = rewriter.create<arith::ConstantFloatOp>(
          loc, rewriter.getF32Type(), llvm::APFloat(*lhsConst));
      if (emitInlineInitOps)
        rewriter.create<FillTileInitOp>(loc);
      rewriter.create<FillTileOp>(loc, curTmpRegAVal, lhsConstVal);
      if (emitInlineInitOps)
        rewriter.create<PowBinaryTilesInitOp>(loc);
      rewriter.create<PowBinaryTilesOp>(loc, curTmpRegAVal, curDstRegVal,
                                        curDstRegVal);
      return success();
    }

    if (auto logOp = dyn_cast<math::LogOp>(defOp)) {
      if (failed(self(self, logOp.getOperand(), curDstReg, curTmpRegA,
                      curTmpRegB)))
        return failure();
      if (emitInlineInitOps)
        rewriter.create<LogTileInitOp>(loc);
      rewriter.create<LogTileOp>(loc, curDstRegVal);
      return success();
    }

    if (auto expOp = dyn_cast<math::ExpOp>(defOp)) {
      if (failed(self(self, expOp.getOperand(), curDstReg, curTmpRegA,
                      curTmpRegB)))
        return failure();
      if (emitInlineInitOps)
        rewriter.create<ExpTileInitOp>(loc);
      rewriter.create<ExpTileOp>(loc, curDstRegVal);
      return success();
    }

    auto emitBinaryMax = [&](Value lhs, Value rhs) -> LogicalResult {
      bool emitRhsFirst = emitRhsFirstForOperands(lhs, rhs);
      if (emitRhsFirst) {
        if (failed(self(self, rhs, curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
        if (failed(self(self, lhs, curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
      } else {
        if (failed(self(self, lhs, curDstReg, curTmpRegA, curTmpRegB)))
          return failure();
        if (failed(self(self, rhs, curTmpRegA, curTmpRegB, curDstReg)))
          return failure();
      }
      if (emitInlineInitOps)
        rewriter.create<BinaryMaxTileInitOp>(loc);
      rewriter.create<BinaryMaxTileOp>(loc, curDstRegVal, curTmpRegAVal,
                                       curDstRegVal);
      return success();
    };

    if (auto maxOp = dyn_cast<arith::MaximumFOp>(defOp)) {
      if (!allowMaximumFOp)
        return failure();
      return emitBinaryMax(maxOp.getLhs(), maxOp.getRhs());
    }

    if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
      Value lhs;
      Value rhs;
      if (!matchMaxSelect(selectOp, lhs, rhs))
        return failure();
      return emitBinaryMax(lhs, rhs);
    }

    return failure();
  };

  return recurse(recurse, exprValue, dstReg, tmpRegA, tmpRegB);
}

static LogicalResult analyzeElementwiseExpr(Value exprValue, linalg::GenericOp op,
                                            ElementwiseAnalysis &analysis) {
  auto handleLeaf = [&](Value value) -> bool {
    auto inputIdx = getBodyInputIndex(op, value);
    if (!inputIdx)
      return false;
    analysis.usedInputs.set(*inputIdx);
    return true;
  };
  auto handleConst = [&]() {};
  auto handlePowBaseConst = [&]() {};
  auto setFlag = [&](GenericExprFeature feature) {
    switch (feature) {
    case GenericExprFeature::BinopWithScalar:
      analysis.needsBinopWithScalar = true;
      return;
    case GenericExprFeature::SubBinary:
      analysis.needsSubBinary = true;
      return;
    case GenericExprFeature::AddBinary:
      analysis.needsAddBinary = true;
      return;
    case GenericExprFeature::MulBinary:
      analysis.needsMulBinary = true;
      return;
    case GenericExprFeature::BinaryMax:
      analysis.needsBinaryMax = true;
      return;
    case GenericExprFeature::PowBinary:
      analysis.needsPowBinary = true;
      return;
    case GenericExprFeature::Recip:
      analysis.needsRecip = true;
      return;
    case GenericExprFeature::Log:
      analysis.needsLog = true;
      return;
    case GenericExprFeature::Exp:
      analysis.needsExp = true;
      return;
    case GenericExprFeature::Fill:
      return;
    }
  };
  return analyzeGenericExprTree(
      exprValue, op, handleLeaf, handleConst, handlePowBaseConst, setFlag,
      /*allowMaximumFOp=*/false);
}

static LogicalResult analyzeElementwiseGeneric(linalg::GenericOp op,
                                               ElementwiseAnalysis &analysis) {
  if (op.getNumDpsInits() != 1)
    return failure();

  for (utils::IteratorType iterType : op.getIteratorTypesArray())
    if (iterType == utils::IteratorType::reduction)
      return failure();

  unsigned rank = op.getNumLoops();
  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  if (!outType || !outType.hasStaticShape() || outType.getRank() != rank)
    return failure();

  auto outTileInfo = getElementwiseTileShapeInfo(outType);
  if (!outTileInfo)
    return failure();
  ArrayRef<int64_t> outTileShape = outTileInfo->dimExtents;

  unsigned numInputs = op.getNumDpsInputs();
  analysis.broadcastsByInput.assign(numInputs, std::nullopt);

  bool allIdentity = hasAllIdentityMapsForRank(op, rank);
  if (!allIdentity) {
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != numInputs + 1)
      return failure();
    if (!isIdentityMapForRank(maps[numInputs], rank))
      return failure();

    for (unsigned i = 0; i < numInputs; ++i) {
      if (isIdentityMapForRank(maps[i], rank))
        continue;

      ElementwiseBroadcastInfo info;
      info.inputIdx = i;
      if (isScalarInputMapForRank(maps[i], rank)) {
        // Scalar map: one scalar tile reused for every output tile.
        info.isScalar = true;
        info.kind = ElementwiseBroadcastKind::BatchBroadcast;
        info.suffixTiles = 1;
        info.droppedDimTiles = outTileInfo->totalTiles;
      } else {
        auto droppedDim = getDroppedDimFromBroadcastMap(maps[i], rank);
        if (!droppedDim)
          return failure();

        auto suffixTiles = getSuffixTilesAfterDim(outTileShape, *droppedDim);
        if (!suffixTiles || *suffixTiles <= 0)
          return failure();

        int64_t droppedDimTiles = outTileShape[*droppedDim];
        if (droppedDimTiles <= 0)
          return failure();

        info.droppedDim = *droppedDim;
        info.kind = classifyElementwiseBroadcastKind(*droppedDim, rank);
        info.droppedDimTiles = droppedDimTiles;
        info.suffixTiles = *suffixTiles;
        // Intra-tile row/col broadcast is lowered by loom.broadcast. Keep
        // linalg.generic focused on scalar and tile-domain batch broadcasts.
        if (info.kind != ElementwiseBroadcastKind::BatchBroadcast)
          return failure();
      }
      analysis.broadcastsByInput[i] = info;
    }

    // Non-identity elementwise maps must all be supported pure broadcasts.
    bool foundBroadcast = llvm::any_of(analysis.broadcastsByInput, [](const auto &info) {
      return info.has_value();
    });
    if (!foundBroadcast)
      return failure();
  }

  // Identity-mapped inputs can still be logical broadcasts when produced by
  // loom.broadcast. Use producer metadata to recover dropped dimension info.
  for (unsigned i = 0; i < numInputs; ++i) {
    if (analysis.broadcastsByInput[i])
      continue;

    auto producerDim = getProducerBroadcastDim(op.getDpsInputs()[i]);
    if (!producerDim)
      continue;
    if (*producerDim >= rank)
      return failure();

    auto suffixTiles = getSuffixTilesAfterDim(outTileShape, *producerDim);
    if (!suffixTiles || *suffixTiles <= 0)
      return failure();

    int64_t droppedDimTiles = outTileShape[*producerDim];
    if (droppedDimTiles <= 0)
      return failure();

    ElementwiseBroadcastInfo info;
    info.inputIdx = i;
    info.isScalar = false;
    info.droppedDim = *producerDim;
    info.kind = classifyElementwiseBroadcastKind(*producerDim, rank);
    info.droppedDimTiles = droppedDimTiles;
    info.suffixTiles = *suffixTiles;
    analysis.broadcastsByInput[i] = info;
  }

  if (outTileInfo->totalTiles <= 0)
    return failure();
  analysis.outTiles = outTileInfo->totalTiles;

  analysis.inputWaitTiles.assign(numInputs, analysis.outTiles);
  analysis.usedInputs.resize(numInputs);
  analysis.usedInputs.reset();

  for (unsigned i = 0; i < numInputs; ++i) {
    if (!analysis.broadcastsByInput[i])
      continue;
    if (analysis.broadcastsByInput[i]->isScalar) {
      analysis.inputWaitTiles[i] = 1;
      continue;
    }
    int64_t droppedDimTiles = analysis.broadcastsByInput[i]->droppedDimTiles;
    if (droppedDimTiles <= 0 || analysis.outTiles % droppedDimTiles != 0)
      return failure();
    analysis.inputWaitTiles[i] = analysis.outTiles / droppedDimTiles;
  }

  for (auto [idx, input] : llvm::enumerate(op.getDpsInputs())) {
    auto annotatedTiles = getAnnotatedVecTilesFromInput(input);
    if (!annotatedTiles)
      continue;
    if (*annotatedTiles <= 0)
      return failure();
    analysis.inputWaitTiles[idx] = *annotatedTiles;
  }

  auto yieldOp = dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return failure();
  analysis.yieldValue = yieldOp.getValues().front();

  if (failed(analyzeElementwiseExpr(analysis.yieldValue, op, analysis)))
    return failure();

  return success();
}

static LogicalResult emitElementwiseExprToReg(
    Value exprValue, int dstReg, int tmpRegA, int tmpRegB, linalg::GenericOp op,
    linalg::GenericOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
    Location loc, ArrayRef<Value> inputCbs, Value tileIdx,
    ArrayRef<std::optional<Value>> broadcastTileIdxByInput,
    ArrayRef<std::optional<Value>> runtimeScalarBitsByInput) {
  auto emitLeaf = [&](Value value, int reg, bool &handled) -> LogicalResult {
    auto inputIdx = getBodyInputIndex(op, value);
    if (!inputIdx) {
      handled = false;
      return success();
    }

    handled = true;
    Value regVal = i32Const(rewriter, loc, reg);
    if (*inputIdx < runtimeScalarBitsByInput.size() &&
        runtimeScalarBitsByInput[*inputIdx].has_value())
      return failure();

    if (*inputIdx < broadcastTileIdxByInput.size() &&
        broadcastTileIdxByInput[*inputIdx].has_value()) {
      Value bcastTileIdx = *broadcastTileIdxByInput[*inputIdx];
      rewriter.create<CopyTileOp>(loc, inputCbs[*inputIdx],
                                  bcastTileIdx, regVal);
      return success();
    }

    rewriter.create<CopyTileOp>(loc, inputCbs[*inputIdx], tileIdx,
                                regVal);
    return success();
  };

  auto emitRhsFirstForOperands = [&](Value lhs, Value rhs) -> bool {
    (void)lhs;
    (void)rhs;
    return false;
  };
  auto lookupRuntimeScalarBits = [&](Value value) -> Value {
    auto inputIdx = getBodyInputIndex(op, value);
    if (!inputIdx || *inputIdx >= runtimeScalarBitsByInput.size() ||
        !runtimeScalarBitsByInput[*inputIdx].has_value())
      return {};
    return *runtimeScalarBitsByInput[*inputIdx];
  };

  return emitGenericExprToRegImpl(
      exprValue, dstReg, tmpRegA, tmpRegB, op, adaptor, rewriter, loc,
      /*emitInlineInitOps=*/false, /*allowMaximumFOp=*/false, emitLeaf,
      emitRhsFirstForOperands, lookupRuntimeScalarBits);
}

static LogicalResult rewriteReduceGeneric(linalg::GenericOp op,
                                          linalg::GenericOp::Adaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          llvm::DenseMap<Value, int64_t> &waitState) {
  if (adaptor.getInputs().size() != 2 || adaptor.getOutputs().size() != 1)
    return failure();

  ReduceType reduceType;
  if (bodyHasOp<arith::MaximumFOp>(op)) {
    reduceType = ReduceType::Max;
  } else if (bodyHasOp<arith::AddFOp>(op)) {
    reduceType = ReduceType::Sum;
  } else {
    return failure();
  }

  Value inCb = adaptor.getInputs()[0];
  Value outCb = adaptor.getOutputs()[0];
  Value scaleCb = adaptor.getInputs()[1];
  if (!isa<CBType>(inCb.getType()) || !isa<CBType>(scaleCb.getType()) ||
      !isa<CBType>(outCb.getType()))
    return failure();

  auto inType = dyn_cast<ShapedType>(op.getDpsInputs()[0].getType());
  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  if (!inType || !outType || !inType.hasStaticShape() ||
      !outType.hasStaticShape() || inType.getRank() != 3)
    return failure();
  if (outType.getRank() != 2 && outType.getRank() != 3)
    return failure();
  if (outType.getRank() == 3 && outType.getShape()[2] != 1)
    return failure();

  // Rank-3 reduction input uses [batch, m, n] where batch is already
  // tile-domain multiplicity (not an element-domain extent).
  int64_t bTiles = inType.getShape()[0];
  auto mTiles = getTileDim(inType, 1);
  auto nTiles = getTileDim(inType, 2);
  auto outTiles = getNumTilesFromShapedType(outType);
  if (bTiles <= 0 || !mTiles || !nTiles || !outTiles)
    return failure();

  int64_t rows = bTiles * (*mTiles);
  int64_t cols = *nTiles;
  int64_t numTiles = rows * cols;
  if (*outTiles != rows)
    return failure();

  Location loc = op.getLoc();
  Value outTilesV = i32Const(rewriter, loc, *outTiles);
  Value zeroI32 = i32Const(rewriter, loc, 0);
  Value oneI32 = i32Const(rewriter, loc, 1);

  SmallVector<std::tuple<Value, Value, int64_t>, 2> inputPlans;
  inputPlans.emplace_back(inCb, op.getDpsInputs()[0], numTiles);
  inputPlans.emplace_back(scaleCb, op.getDpsInputs()[1], 1);

  for (const auto &plan : inputPlans) {
    Value inputCb = std::get<0>(plan);
    int64_t waitTiles = std::get<2>(plan);
    emitWaitFrontIfNeeded(rewriter, loc, inputCb, waitTiles, waitState);
  }

  CBReserveBackOp::create(rewriter, loc, outCb, outTilesV);
  Value rowsV = i32Const(rewriter, loc, rows);
  Value colsV = i32Const(rewriter, loc, cols);
  scf::ForOp rowLoop =
      scf::ForOp::create(rewriter, loc, zeroI32, rowsV, oneI32);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(rowLoop.getBody());

    Value rowIdx = rowLoop.getInductionVar();
    TileRegsAcquireOp::create(rewriter, loc);
    rewriter.create<ReduceInitOp>(loc, inCb, scaleCb, outCb, reduceType,
                                  ReduceDim::Row);

    Value rowOffset = rewriter.create<arith::MulIOp>(loc, rowIdx, colsV);
    scf::ForOp colLoop =
        scf::ForOp::create(rewriter, loc, zeroI32, colsV, oneI32);
    {
      OpBuilder::InsertionGuard innerGuard(rewriter);
      rewriter.setInsertionPointToStart(colLoop.getBody());
      Value colIdx = colLoop.getInductionVar();
      Value inTile = rewriter.create<arith::AddIOp>(loc, rowOffset, colIdx);
      rewriter.create<ReduceTileOp>(loc, inCb, scaleCb, inTile, zeroI32,
                                    zeroI32, reduceType, ReduceDim::Row);
    }

    rewriter.setInsertionPointAfter(colLoop);
    rewriter.create<ReduceUninitOp>(loc);
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);
    PackTileOp::create(rewriter, loc, zeroI32, outCb, rowIdx);
    TileRegsReleaseOp::create(rewriter, loc);
  }
  //TODO: add pop first, since current output also works as input
  //CBPopFrontOp::create(rewriter, loc, outCb, outTilesV);
  //CBReserveBackOp::create(rewriter, loc, outCb, outTilesV);
  CBPushBackOp::create(rewriter, loc, outCb, outTilesV);
  waitState[outCb] = 0;

  rewriter.eraseOp(op);
  return success();
}

static LogicalResult rewriteElementwiseGeneric(linalg::GenericOp op,
                                               linalg::GenericOp::Adaptor adaptor,
                                               ConversionPatternRewriter &rewriter,
                                               llvm::DenseMap<Value, int64_t> &waitState,
                                               std::shared_ptr<CompileArgTracker> tracker) {
  if (adaptor.getOutputs().size() != 1 || adaptor.getInputs().empty())
    return failure();

  ElementwiseAnalysis analysis;
  if (failed(analyzeElementwiseGeneric(op, analysis)))
    return failure();

  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(outCb.getType()))
    return failure();
  SmallVector<Value, 4> inputCbs;
  inputCbs.reserve(adaptor.getInputs().size());
  for (Value inputCb : adaptor.getInputs()) {
    Value effectiveInputCb = stripBroadcastBridgeCast(inputCb);
    if (!isa<CBType>(effectiveInputCb.getType()))
      return failure();
    inputCbs.push_back(effectiveInputCb);
  }
  for (Value inputCb : inputCbs)
    if (!isa<CBType>(inputCb.getType()))
      return failure();

  Location loc = op.getLoc();
  bool outAliasesInput = hasOutputAlias(outCb, inputCbs);
  unsigned numInputs = adaptor.getInputs().size();
  SmallVector<std::optional<Value>, 4> runtimeScalarBitsByInput(numInputs,
                                                                 std::nullopt);
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (tracker && parentFunc) {
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i >= analysis.broadcastsByInput.size() || !analysis.broadcastsByInput[i] ||
          !analysis.broadcastsByInput[i]->isScalar)
        continue;
      if (!analysis.usedInputs.test(i))
        continue;

      auto siteId = getScalarSiteIdForGenericInput(op, i);
      if (!siteId)
        continue;
      Value scalarBits =
          tracker->getScalarRuntimeArg(parentFunc.getOperation(), *siteId);
      if (!scalarBits)
        continue;
      runtimeScalarBitsByInput[i] = scalarBits;
      analysis.needsBinopWithScalar = true;
    }
  }

  for (auto [idx, inputCb] : llvm::enumerate(inputCbs)) {
    if (!analysis.usedInputs.test(idx))
      continue;
    if (idx < runtimeScalarBitsByInput.size() &&
        runtimeScalarBitsByInput[idx].has_value())
      continue;

    int64_t waitTiles = analysis.inputWaitTiles.empty()
                            ? analysis.outTiles
                            : analysis.inputWaitTiles[idx];
    emitWaitFrontIfNeeded(rewriter, loc, inputCb, waitTiles, waitState);

    if (inputCb == outCb)
      continue;
  }

  if (!outAliasesInput)
    CBReserveBackOp::create(rewriter, loc, outCb,
                            i32Const(rewriter, loc, analysis.outTiles));

  Value inCbForInit = outCb;
  for (auto [idx, inputCb] : llvm::enumerate(inputCbs)) {
    if (analysis.usedInputs.test(idx) &&
        !(idx < runtimeScalarBitsByInput.size() &&
          runtimeScalarBitsByInput[idx].has_value())) {
      inCbForInit = inputCb;
      break;
    }
  }
  rewriter.create<InitSFPUOp>(loc, inCbForInit, outCb);
  for (auto [idx, inputCb] : llvm::enumerate(inputCbs)) {
    if (!analysis.usedInputs.test(idx))
      continue;
    if (idx < runtimeScalarBitsByInput.size() &&
        runtimeScalarBitsByInput[idx].has_value())
      continue;
    rewriter.create<CopyTileInitOp>(loc, inputCb);
  }
  if (analysis.needsBinopWithScalar)
    rewriter.create<BinopWithScalarTileInitOp>(loc);
  if (analysis.needsSubBinary)
    rewriter.create<SubBinaryTilesInitOp>(loc);
  if (analysis.needsAddBinary)
    rewriter.create<AddBinaryTilesInitOp>(loc);
  if (analysis.needsMulBinary)
    rewriter.create<MulBinaryTilesInitOp>(loc);
  if (analysis.needsBinaryMax)
    rewriter.create<BinaryMaxTileInitOp>(loc);
  if (analysis.needsPowBinary) {
    rewriter.create<FillTileInitOp>(loc);
    rewriter.create<PowBinaryTilesInitOp>(loc);
  }
  if (analysis.needsRecip)
    rewriter.create<RecipTileInitOp>(loc);
  if (analysis.needsLog)
    rewriter.create<LogTileInitOp>(loc);
  if (analysis.needsExp)
    rewriter.create<ExpTileInitOp>(loc);

  LogicalResult result = emitElementwiseTiles(
      rewriter, loc, outCb, analysis.outTiles, [&](Value tileIdx) -> LogicalResult {
        SmallVector<std::optional<Value>, 4> broadcastTileIdxByInput(
            numInputs, std::nullopt);
        for (unsigned i = 0; i < numInputs; ++i) {
          if (i >= analysis.broadcastsByInput.size() || !analysis.broadcastsByInput[i])
            continue;
          const ElementwiseBroadcastInfo &broadcastInfo =
              *analysis.broadcastsByInput[i];
          if (broadcastInfo.isScalar) {
            broadcastTileIdxByInput[i] = i32Const(rewriter, loc, 0);
            continue;
          }
          Value suffixTiles = i32Const(rewriter, loc, broadcastInfo.suffixTiles);
          Value droppedDimSpan = i32Const(
              rewriter, loc,
              broadcastInfo.droppedDimTiles * broadcastInfo.suffixTiles);
          Value prefix = rewriter.create<arith::DivSIOp>(loc, tileIdx, droppedDimSpan);
          if (broadcastInfo.suffixTiles == 1) {
            broadcastTileIdxByInput[i] = prefix;
          } else {
            Value suffixOffset =
                rewriter.create<arith::RemSIOp>(loc, tileIdx, suffixTiles);
            Value prefixBase =
                rewriter.create<arith::MulIOp>(loc, prefix, suffixTiles);
            broadcastTileIdxByInput[i] =
                rewriter.create<arith::AddIOp>(loc, prefixBase, suffixOffset);
          }
        }
        return emitElementwiseExprToReg(
            analysis.yieldValue, /*dstReg=*/0, /*tmpRegA=*/1, /*tmpRegB=*/2, op,
            adaptor, rewriter, loc, inputCbs, tileIdx, broadcastTileIdxByInput,
            runtimeScalarBitsByInput);
      });
  if (failed(result))
    return failure();

  if (outAliasesInput) {
    //release the input cb first for later usage of output cb
    CBPopFrontOp::create(rewriter, loc, outCb,
                         i32Const(rewriter, loc, analysis.outTiles));
    CBReserveBackOp::create(rewriter, loc, outCb,
                            i32Const(rewriter, loc, analysis.outTiles));
    CBPushBackOp::create(rewriter, loc, outCb,
                         i32Const(rewriter, loc, analysis.outTiles));
    waitState[outCb] = 0;
  } else {
    CBPushBackOp::create(rewriter, loc, outCb,
                         i32Const(rewriter, loc, analysis.outTiles));
    waitState[outCb] = 0;
  }

  rewriter.eraseOp(op);
  return success();
}

/**
 * @brief Lower `linalg.batch_matmul` by reusing TTKernel matmul block ops.
 *
 * @details This pattern keeps the existing matmul-block execution model and
 *          iterates over the batch dimension by advancing input tile offsets.
 *          It reserves/pushes the whole output CB once and packs each batch's
 *          tile-register result range into the corresponding output tile slice.
 */
class ConvertLinalgBatchMatmulOp
    : public OpConversionPattern<linalg::BatchMatmulOp> {
public:
  using OpConversionPattern<linalg::BatchMatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::BatchMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInputs().size() != 2 || adaptor.getOutputs().size() != 1)
      return failure();

    Location loc = op.getLoc();
    Value in0Cb = adaptor.getInputs()[0];
    Value in1Cb = adaptor.getInputs()[1];
    Value outCb = adaptor.getOutputs()[0];

    if (!isa<CBType>(in0Cb.getType()) || !isa<CBType>(in1Cb.getType()) ||
        !isa<CBType>(outCb.getType()))
      return failure();

    auto lhsShapedType = dyn_cast<ShapedType>(op.getInputs()[0].getType());
    auto rhsShapedType = dyn_cast<ShapedType>(op.getInputs()[1].getType());
    auto outShapedType = dyn_cast<ShapedType>(op.getOutputs()[0].getType());
    std::optional<MatmulTileInfo> tileInfo =
        getMatmulTileInfo(lhsShapedType, rhsShapedType, outShapedType);
    if (!tileInfo || tileInfo->batchSize <= 0)
      return failure();

    if (lhsShapedType.getRank() != 3 || rhsShapedType.getRank() != 3 ||
        outShapedType.getRank() != 3) {
      return failure();
    }

    Value zeroI32 = i32Const(rewriter, loc, 0);
    Value oneI32 = i32Const(rewriter, loc, 1);
    Value transpose = zeroI32;
    Value ctDim = i32Const(rewriter, loc, tileInfo->ct);
    Value rtDim = i32Const(rewriter, loc, tileInfo->rt);
    Value ntDim = i32Const(rewriter, loc, tileInfo->nt);
    Value ktDim = i32Const(rewriter, loc, tileInfo->kt);

    rewriter.create<MatmulBlockInitShortOp>(
        loc, TypeRange{}, ValueRange{in0Cb, in1Cb, transpose, ctDim, rtDim, ktDim});

    Value in0TileCount = i32Const(rewriter, loc, tileInfo->in0TilesTotal);
    Value in1TileCount = i32Const(rewriter, loc, tileInfo->in1TilesTotal);
    if (in0Cb == in1Cb) {
      int64_t sharedTiles =
          std::max(tileInfo->in0TilesTotal, tileInfo->in1TilesTotal);
      CBWaitFrontOp::create(rewriter, loc, in0Cb,
                            i32Const(rewriter, loc, sharedTiles));
    } else {
      CBWaitFrontOp::create(rewriter, loc, in0Cb, in0TileCount);
      CBWaitFrontOp::create(rewriter, loc, in1Cb, in1TileCount);
    }

    Value totalOutTiles = i32Const(rewriter, loc, tileInfo->outTilesTotal);
    Value batchTiles = i32Const(rewriter, loc, tileInfo->outTilesPerBatch);
    Value in0BatchStride = i32Const(rewriter, loc, tileInfo->in0TilesPerBatch);
    Value in1BatchStride = i32Const(rewriter, loc, tileInfo->in1TilesPerBatch);
    Value batchCount = i32Const(rewriter, loc, tileInfo->batchSize);
    CBReserveBackOp::create(rewriter, loc, outCb, totalOutTiles);

    scf::ForOp batchLoop =
        scf::ForOp::create(rewriter, loc, zeroI32, batchCount, oneI32);
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    Value batchIdx = batchLoop.getInductionVar();
    Value in0TileIdx = rewriter.create<arith::MulIOp>(loc, batchIdx, in0BatchStride);
    Value in1TileIdx = rewriter.create<arith::MulIOp>(loc, batchIdx, in1BatchStride);
    Value outTileBase = rewriter.create<arith::MulIOp>(loc, batchIdx, batchTiles);

    TileRegsAcquireOp::create(rewriter, loc);
    rewriter.create<ExperimentalMatmulBlockOp>(
        loc, TypeRange{},
        ValueRange{in0Cb, in1Cb, in0TileIdx, in1TileIdx, zeroI32, transpose,
                   ctDim, rtDim, ktDim, ntDim});
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);

    scf::ForOp packLoop =
        scf::ForOp::create(rewriter, loc, zeroI32, batchTiles, oneI32);
    rewriter.setInsertionPointToStart(packLoop.getBody());
    Value i = packLoop.getInductionVar();
    Value outTileIdx = rewriter.create<arith::AddIOp>(loc, outTileBase, i);
    PackTileOp::create(rewriter, loc, i, outCb, outTileIdx);
    rewriter.setInsertionPointAfter(packLoop);

    TileRegsReleaseOp::create(rewriter, loc);
    rewriter.setInsertionPointAfter(batchLoop);

    CBPushBackOp::create(rewriter, loc, outCb, totalOutTiles);
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertLinalgFillOp : public OpConversionPattern<linalg::FillOp> {
public:
  using OpConversionPattern<linalg::FillOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInputs().size() != 1 || adaptor.getOutputs().size() != 1)
      return failure();

    Value outCb = adaptor.getOutputs()[0];
    if (!isa<CBType>(outCb.getType()))
      return failure();

    // If the fill's output buffer is later used as an output of another
    // linalg op in the same block (and not also as that op's input),
    // the fill is redundant and can be erased.
    if (auto nextUse = findNextNonViewUseInSameBlock(op.getDpsInits()[0],
                                                     op.getOperation())) {
      auto linalgUser = dyn_cast<linalg::LinalgOp>(nextUse->user);
      if (linalgUser) {
        bool isOutput =
            llvm::is_contained(linalgUser.getDpsInits(), nextUse->usedValue);
        bool isInput =
            llvm::is_contained(linalgUser.getDpsInputs(), nextUse->usedValue);
        if (isOutput && !isInput) {
          rewriter.eraseOp(op);
          return success();
        }
      }
    }

    auto numTiles = getNumTilesFromShapedType(op.getDpsInits()[0].getType());
    if (!numTiles)
      return failure();

    Location loc = op.getLoc();
    Value fillValue = adaptor.getInputs()[0];
    if (!fillValue.getType().isF32()) {
      if (fillValue.getType().isIntOrIndex()) {
        fillValue =
            rewriter.create<arith::SIToFPOp>(loc, rewriter.getF32Type(), fillValue);
      } else if (llvm::isa<FloatType>(fillValue.getType())) {
        fillValue =
            rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), fillValue);
      } else {
        return failure();
      }
    }

    Value zeroI32 = i32Const(rewriter, loc, 0);
    Value oneI32 = i32Const(rewriter, loc, 1);
    Value numTilesV = i32Const(rewriter, loc, *numTiles);
    CBReserveBackOp::create(rewriter, loc, outCb, numTilesV);
    rewriter.create<InitSFPUOp>(loc, outCb, outCb);
    rewriter.create<FillTileInitOp>(loc);

    scf::ForOp fillLoop =
        scf::ForOp::create(rewriter, loc, zeroI32, numTilesV, oneI32);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(fillLoop.getBody());
      Value tileIdx = fillLoop.getInductionVar();
      TileRegsAcquireOp::create(rewriter, loc);
      rewriter.create<FillTileOp>(loc, zeroI32, fillValue);
      TileRegsCommitOp::create(rewriter, loc);
      TileRegsWaitOp::create(rewriter, loc);
      PackTileOp::create(rewriter, loc, zeroI32, outCb, tileIdx);
      TileRegsReleaseOp::create(rewriter, loc);
    }

    CBPushBackOp::create(rewriter, loc, outCb, numTilesV);
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertLinalgCopyOp : public OpConversionPattern<linalg::CopyOp> {
public:
  using OpConversionPattern<linalg::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!mlir::loom::shouldConvertComputeLinalgCopy(op))
      return failure();

    if (adaptor.getInputs().size() != 1 || adaptor.getOutputs().size() != 1)
      return failure();

    Value inCb = adaptor.getInputs()[0];
    Value outCb = adaptor.getOutputs()[0];
    if (!isa<CBType>(inCb.getType()) || !isa<CBType>(outCb.getType()))
      return failure();

    auto inTiles = getNumTilesFromShapedType(op.getInputs()[0].getType());
    auto outTiles = getNumTilesFromShapedType(op.getOutputs()[0].getType());
    if (!inTiles || !outTiles || *inTiles != *outTiles)
      return failure();

    if (inCb == outCb) {
      rewriter.eraseOp(op);
      return success();
    }

    Location loc = op.getLoc();
    Value tileCount = i32Const(rewriter, loc, *inTiles);

    CBPopFrontOp::create(rewriter, loc, outCb, tileCount);
    // Preserve current linalg.copy ownership semantics:
    // - destination window is explicitly rotated (pop+push here)
    // - source consumption is handled elsewhere (no cb_pop_front on inCb here)
    copyTile(rewriter, loc, inCb, outCb, tileCount, /*popInputCb=*/false);
    CBPushBackOp::create(rewriter, loc, outCb, tileCount);
    rewriter.eraseOp(op);
    return success();
  }
};

static bool has2DSwapPermutation(linalg::TransposeOp op) {
  Attribute permAttr = op->getAttr("permutation");
  if (auto dense = dyn_cast_or_null<DenseI64ArrayAttr>(permAttr)) {
    if (dense.size() != 2)
      return false;
    return dense[0] == 1 && dense[1] == 0;
  }
  if (auto arr = dyn_cast_or_null<ArrayAttr>(permAttr)) {
    if (arr.size() != 2)
      return false;
    auto p0 = dyn_cast<IntegerAttr>(arr[0]);
    auto p1 = dyn_cast<IntegerAttr>(arr[1]);
    if (!p0 || !p1)
      return false;
    return p0.getInt() == 1 && p1.getInt() == 0;
  }
  return false;
}

class ConvertLinalgTransposeOp : public OpConversionPattern<linalg::TransposeOp> {
public:
  using OpConversionPattern<linalg::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!mlir::loom::shouldConvertComputeLinalgTranspose(op))
      return failure();

    Value inCb = adaptor.getInput();
    Value outCb = adaptor.getInit();
    if (!isa<CBType>(inCb.getType()) || !isa<CBType>(outCb.getType()))
      return failure();

    auto inTiles = getNumTilesFromShapedType(op.getInput().getType());
    auto outTiles = getNumTilesFromShapedType(op.getInit().getType());
    if (!inTiles || !outTiles || *inTiles != *outTiles)
      return failure();

    if (inCb == outCb) {
      rewriter.eraseOp(op);
      return success();
    }

    Location loc = op.getLoc();
    Value tileCount = i32Const(rewriter, loc, *inTiles);

    // Reuse copyTile transport path as an initial transpose lowering entry.
    // linalg.transpose is typically followed by loom.semaphore_give on the
    // source buffer; let that lowering own the cb_pop_front to avoid
    // double-pop mismatches.
    copyTile(rewriter, loc, inCb, outCb, tileCount, /*popInputCb=*/false);
    CBPushBackOp::create(rewriter, loc, outCb, tileCount);

    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertLoomSyncOp : public OpConversionPattern<::loom::SyncOp> {
public:
  using OpConversionPattern<::loom::SyncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::loom::SyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!mlir::loom::shouldConvertComputeLoomSync(op))
      return failure();

    if (adaptor.getOperands().size() != 2)
      return failure();

    Value inCb = adaptor.getOperands()[0];
    Value outCb = adaptor.getOperands()[1];
    if (!isa<CBType>(inCb.getType()) || !isa<CBType>(outCb.getType()))
      return failure();

    // Memref-form loom.sync has no result value to replace.
    if (op.getNumResults() != 0)
      return failure();

    auto inTiles = getNumTilesFromShapedType(op.getIns().getType());
    auto outTiles = getNumTilesFromShapedType(op.getInit().getType());
    if (!inTiles || !outTiles || *inTiles != *outTiles)
      return failure();

    // Synchronization source/destination may alias the same CB handle. In that
    // case, no transport is needed.
    if (inCb == outCb) {
      rewriter.eraseOp(op);
      return success();
    }

    Location loc = op.getLoc();
    Value tileCount = i32Const(rewriter, loc, *inTiles);

    // Keep input ownership with semaphore_give lowering to avoid double-pop.
    //[IMPORTANT] it seems hangs for mamba chunk scan, not safe
    //CBPopFrontOp::create(rewriter, loc, inCb, tileCount);
    //copyTile(rewriter, loc, inCb, outCb, tileCount, /*popInputCb=*/false);
    CBWaitFrontOp::create(rewriter, loc, inCb, tileCount);
    CBPushBackOp::create(rewriter, loc, outCb, tileCount);

    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertLoomBroadcastOp : public OpConversionPattern<::loom::BroadcastOp> {
public:
  using OpConversionPattern<::loom::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::loom::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!mlir::loom::shouldConvertComputeLoomBroadcast(op))
      return failure();
    if (adaptor.getOperands().size() != 2)
      return failure();

    Value inCb = adaptor.getOperands()[0];
    Value outCb = adaptor.getOperands()[1];
    if (!isa<CBType>(inCb.getType()) || !isa<CBType>(outCb.getType()))
      return failure();

    auto inTiles = getNumTilesFromShapedType(op.getIns().getType());
    auto outTiles = getNumTilesFromShapedType(op.getInit().getType());
    if (!inTiles || !outTiles || *inTiles != *outTiles)
      return failure();

    unsigned rank = 0;
    if (op.getNumResults() > 0) {
      if (op.getNumResults() != 1)
        return failure();
      auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
      if (!resultType || !resultType.hasStaticShape())
        return failure();
      rank = resultType.getRank();
    } else {
      auto outType = dyn_cast<ShapedType>(op.getInit().getType());
      if (!outType || !outType.hasStaticShape())
        return failure();
      rank = outType.getRank();
    }

    int64_t dim = op.getDim();
    if (dim < 0 || static_cast<unsigned>(dim) >= rank)
      return failure();

    auto kind = classifyElementwiseBroadcastKind(static_cast<unsigned>(dim), rank);
    std::optional<BcastType> bcastType = getUnaryBcastType(kind);
    auto finalizeReplacement = [&]() -> LogicalResult {
      if (op.getNumResults() == 0) {
        rewriter.eraseOp(op);
        return success();
      }

      Type convertedResultType =
          getTypeConverter()->convertType(op->getResult(0).getType());
      if (!convertedResultType)
        return failure();
      if (convertedResultType == outCb.getType()) {
        rewriter.replaceOp(op, ValueRange{outCb});
        return success();
      }

      auto cast = rewriter.create<UnrealizedConversionCastOp>(
          op.getLoc(), TypeRange{convertedResultType}, outCb);
      cast->setAttr(kBroadcastDimAttrName, rewriter.getI64IntegerAttr(dim));
      rewriter.replaceOp(op, cast.getResults());
      return success();
    };

    if (inCb == outCb && !bcastType)
      return finalizeReplacement();
    if (inCb == outCb && bcastType)
      return failure();

    Location loc = op.getLoc();
    Value tileCount = i32Const(rewriter, loc, *inTiles);
    Value zero = i32Const(rewriter, loc, 0);
    Value one = i32Const(rewriter, loc, 1);
    CBWaitFrontOp::create(rewriter, loc, inCb, tileCount);
    if (inCb != outCb)
      CBReserveBackOp::create(rewriter, loc, outCb, tileCount);

    if (bcastType)
      rewriter.create<UnaryBcastInitOp>(loc, inCb, outCb, *bcastType);
    else
      rewriter.create<CopyTileInitOp>(loc, inCb);

    scf::ForOp tileLoop =
        scf::ForOp::create(rewriter, loc, zero, tileCount, one);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(tileLoop.getBody());
      Value tileIdx = tileLoop.getInductionVar();
      TileRegsAcquireOp::create(rewriter, loc);
      if (bcastType) {
        rewriter.create<UnaryBcastTileOp>(loc, inCb, tileIdx, zero, *bcastType);
      } else {
        rewriter.create<CopyTileOp>(loc, inCb, tileIdx, zero);
      }
      TileRegsCommitOp::create(rewriter, loc);
      TileRegsWaitOp::create(rewriter, loc);
      if (inCb != outCb)
        PackTileOp::create(rewriter, loc, zero, outCb, tileIdx);
      TileRegsReleaseOp::create(rewriter, loc);
    }

    if (inCb != outCb) {
      //CBPopFrontOp::create(rewriter, loc, inCb, tileCount);
      CBPushBackOp::create(rewriter, loc, outCb, tileCount);
    }

    return finalizeReplacement();
  }
};

class ConvertMemrefCollapseShapeOp
    : public OpConversionPattern<memref::CollapseShapeOp> {
public:
  using OpConversionPattern<memref::CollapseShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CollapseShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<MemRefType>(op.getSrcType());
    auto dstTy = dyn_cast<MemRefType>(op.getResultType());
    if (!srcTy || !dstTy || !srcTy.hasStaticShape() || !dstTy.hasStaticShape())
      return failure();
    if (srcTy.getNumElements() != dstTy.getNumElements())
      return failure();

    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

class ConvertMemrefExpandShapeOp
    : public OpConversionPattern<memref::ExpandShapeOp> {
public:
  using OpConversionPattern<memref::ExpandShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExpandShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<MemRefType>(op.getSrcType());
    auto dstTy = dyn_cast<MemRefType>(op.getResultType());
    if (!srcTy || !dstTy)
      return failure();

    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

class ConvertFlashAttentionGenericOp
    : public OpConversionPattern<linalg::GenericOp> {
public:
  ConvertFlashAttentionGenericOp(TypeConverter &typeConverter,
                                 MLIRContext *context,
                                 std::shared_ptr<CompileArgTracker> tracker)
      : OpConversionPattern<linalg::GenericOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
    if (parentFunc != activeFunc) {
      waitState.clear();
      activeFunc = parentFunc;
    }

    // Route [reduction, parallel, parallel] through tileGenericOp first.
    if (isTileGenericOp(op))
      return tileGenericOp(op, adaptor, rewriter, waitState);

    if (!mlir::loom::isSupportedFlashAttentionGeneric(op))
      return failure();

    auto kind = classifyFlashAttentionGeneric(op);
    if (!kind)
      return failure();

    switch (*kind) {
    case FlashAttentionGenericKind::Reduction:
      return rewriteReduceGeneric(op, adaptor, rewriter, waitState);
    case FlashAttentionGenericKind::Elementwise:
      return rewriteElementwiseGeneric(op, adaptor, rewriter, waitState,
                                       tracker);
    }

    return failure();
  }

private:
  mutable llvm::DenseMap<Value, int64_t> waitState;
  mutable func::FuncOp activeFunc;
  std::shared_ptr<CompileArgTracker> tracker;
};

} // namespace

bool mlir::loom::isSupportedFlashAttentionGeneric(linalg::GenericOp op) {
  return isComputeKernel(op.getOperation()) &&
         classifyFlashAttentionGeneric(op).has_value();
}

bool mlir::loom::shouldConvertComputeLinalgCopy(linalg::CopyOp op) {
  if (!isComputeKernel(op.getOperation()))
    return false;

  if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
    return false;

  auto inTiles = getNumTilesFromShapedType(op.getInputs()[0].getType());
  auto outTiles = getNumTilesFromShapedType(op.getOutputs()[0].getType());
  return inTiles && outTiles && *inTiles == *outTiles;
}

bool mlir::loom::shouldConvertComputeLinalgTranspose(linalg::TransposeOp op) {
  if (!isComputeKernel(op.getOperation()))
    return false;

  auto inType = dyn_cast<ShapedType>(op.getInput().getType());
  auto outType = dyn_cast<ShapedType>(op.getInit().getType());
  if (!inType || !outType || !inType.hasStaticShape() || !outType.hasStaticShape())
    return false;

  if (inType.getRank() != 2 || outType.getRank() != 2)
    return false;

  if (!has2DSwapPermutation(op))
    return false;

  auto inTiles = getNumTilesFromShapedType(op.getInput().getType());
  auto outTiles = getNumTilesFromShapedType(op.getInit().getType());
  return inTiles && outTiles && *inTiles == *outTiles;
}

bool mlir::loom::shouldConvertComputeLoomSync(::loom::SyncOp op) {
  if (!isComputeKernel(op.getOperation()))
    return false;

  if (op.getNumResults() != 0)
    return false;

  auto inTiles = getNumTilesFromShapedType(op.getIns().getType());
  auto outTiles = getNumTilesFromShapedType(op.getInit().getType());
  return inTiles && outTiles && *inTiles == *outTiles;
}

bool mlir::loom::shouldConvertComputeLoomBroadcast(::loom::BroadcastOp op) {
  if (!isComputeKernel(op.getOperation()))
    return false;

  auto inTiles = getNumTilesFromShapedType(op.getIns().getType());
  auto outTiles = getNumTilesFromShapedType(op.getInit().getType());
  if (!inTiles || !outTiles || *inTiles != *outTiles)
    return false;

  int64_t dim = op.getDim();
  if (dim < 0)
    return false;

  unsigned rank = 0;
  if (op.getNumResults() > 0) {
    if (op.getNumResults() != 1)
      return false;
    auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return false;
    auto resultTiles = getNumTilesFromShapedType(resultType);
    if (!resultTiles || *resultTiles < *outTiles || *resultTiles % *outTiles != 0)
      return false;
    rank = resultType.getRank();
  } else {
    auto initType = dyn_cast<ShapedType>(op.getInit().getType());
    if (!initType || !initType.hasStaticShape())
      return false;
    rank = initType.getRank();
  }

  return static_cast<unsigned>(dim) < rank;
}

void mlir::loom::populateComputeOpConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context, std::shared_ptr<CompileArgTracker> tracker) {
  patterns.add<ConvertLinalgFillOp>(typeConverter, context);
  patterns.add<ConvertLinalgCopyOp>(typeConverter, context);
  patterns.add<ConvertLinalgTransposeOp>(typeConverter, context);
  patterns.add<ConvertLoomSyncOp>(typeConverter, context);
  patterns.add<ConvertLoomBroadcastOp>(typeConverter, context);
  patterns.add<ConvertLinalgBatchMatmulOp>(typeConverter, context);
  patterns.add<ConvertMemrefCollapseShapeOp>(typeConverter, context);
  patterns.add<ConvertMemrefExpandShapeOp>(typeConverter, context);
  patterns.add<ConvertFlashAttentionGenericOp>(typeConverter, context,
                                               std::move(tracker));
}
