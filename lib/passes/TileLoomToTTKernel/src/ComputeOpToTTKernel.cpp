/**
 * @file ComputeOpToTTKernel.cpp
 * @brief Conversion patterns for compute ops to TTKernel.
 */

#include "ComputeOpToTTKernel.h"
#include "FuncOpToTTKernel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
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
using namespace tt::ttkernel;
using mlir::loom::CompileArgTracker;
using mlir::loom::ReduceCombineOp;
using mlir::loom::ReduceProtocol;

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
  std::optional<ElementwiseBroadcastInfo> broadcast;
  SmallVector<int64_t, 4> inputWaitTiles;
  llvm::SmallBitVector usedInputs;

  bool needsBinopWithScalar = false;
  bool needsUnaryBcast = false;
  bool needsSubBinary = false;
  bool needsAddBinary = false;
  bool needsMulBinary = false;
  bool needsBinaryMax = false;
  bool needsPowBinary = false;
  bool needsRecip = false;
  bool needsLog = false;
};

static std::optional<int64_t> ceilDiv32(int64_t value) {
  if (value <= 0)
    return std::nullopt;
  return (value + 31) / 32;
}

static LogicalResult analyzeElementwiseGeneric(linalg::GenericOp op,
                                               ElementwiseAnalysis &analysis);
static bool isSupportedElementwiseGeneric(linalg::GenericOp op) {
  ElementwiseAnalysis analysis;
  return succeeded(analyzeElementwiseGeneric(op, analysis));
}

static bool isComputeKernel(Operation *op) {
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return false;

  if (auto threadAttr =
          parentFunc->getAttrOfType<ThreadTypeAttr>(ThreadTypeAttr::name)) {
    return threadAttr.getValue() == ThreadType::Compute;
  }

  return parentFunc.getName().ends_with("__compute");
}

/**
 * @brief Collect non-view users reachable from a value through view aliases.
 *
 * @details Walks users of `rootValue` and follows view-like ops (casts/views)
 *          to discover the effective non-view consumers. `excludedUser` is
 *          skipped (used to ignore the producer op currently being rewritten).
 */
static void collectNonViewUsersThroughViews(
    Value rootValue, Operation *excludedUser,
    SmallVectorImpl<Operation *> &users) {
  SmallVector<Value, 8> worklist;
  llvm::SmallPtrSet<Value, 16> seenValues;
  llvm::SmallPtrSet<Operation *, 16> seenUsers;

  worklist.push_back(rootValue);
  seenValues.insert(rootValue);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (Operation *user : current.getUsers()) {
      if (user == excludedUser)
        continue;

      if (isa<ViewLikeOpInterface>(user)) {
        for (Value result : user->getResults())
          if (seenValues.insert(result).second)
            worklist.push_back(result);
        continue;
      }

      if (seenUsers.insert(user).second)
        users.push_back(user);
    }
  }
}

/**
 * @brief Choose where matmul output tile-reg materialization should be placed.
 *
 * @details If a matmul output is consumed in the same block after matmul, the
 *          materialization scope is the matmul itself. If not, but the output
 *          is consumed in an ancestor block, the nearest enclosing op whose
 *          parent block contains that next consumer is returned (e.g. an
 *          enclosing scf.for). This enables wrapping tile-reg acquire/commit
 *          around loop scopes when the next use is outside the loop body.
 */
static Operation *findMatmulOutputMaterializationScope(Value outputBuffer,
                                                       Operation *matmulOp) {
  SmallVector<Operation *, 8> users;
  collectNonViewUsersThroughViews(outputBuffer, matmulOp, users);
  if (users.empty())
    return matmulOp;

  Block *matmulBlock = matmulOp->getBlock();
  for (Operation *user : users) {
    if (user->getBlock() == matmulBlock && matmulOp->isBeforeInBlock(user))
      return matmulOp;
  }

  for (Operation *ancestor = matmulOp->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    Block *ancestorBlock = ancestor->getBlock();
    if (!ancestorBlock)
      continue;
    for (Operation *user : users) {
      if (user->getBlock() == ancestorBlock &&
          ancestor->isBeforeInBlock(user))
        return ancestor;
    }
  }

  return matmulOp;
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

/**
 * @brief Captures TTKernel init-op requirements for one tile-generic body.
 */
struct TileGenericExprAnalysis {
  Value yieldValue;
  bool needsAccumulatorCopy = false;
  bool needsBinopWithScalar = false;
  bool needsFill = false;
  bool needsSubBinary = false;
  bool needsAddBinary = false;
  bool needsMulBinary = false;
  bool needsBinaryMax = false;
  bool needsPowBinary = false;
  bool needsRecip = false;
  bool needsLog = false;
};

/**
 * @brief Distinguishes tile-generic body block arguments by semantic role.
 */
enum class TileGenericBodyOperand {
  Input,
  Accumulator
};

/**
 * @brief Analyze whether a tile generic body expression is lowerable.
 */
static LogicalResult analyzeTileGeneric(linalg::GenericOp op,
                                        TileGenericExprAnalysis &analysis);

/**
 * @brief Emit one tile generic body expression into a destination register.
 */
static LogicalResult emitTileGenericExprToReg(
    Value exprValue, int dstReg, int tmpRegA, int tmpRegB, linalg::GenericOp op,
    linalg::GenericOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
    Location loc, Value inputTileIdx, Value accumulatorTileIdx,
    bool emitInlineInitOps);

/**
 * @brief Check for the [reduction, parallel, parallel] iterator pattern.
 */
static bool isTileGenericOp(linalg::GenericOp op) {
  auto iteratorTypes = op.getIteratorTypesArray();
  if (iteratorTypes.size() < 3)
    return false;
  return iteratorTypes[0] == utils::IteratorType::reduction;
}

/**
 * @brief Entry-point for [reduction, parallel, parallel] generic lowering.
 *
 * @details The first reduction slice seeds the output tile, then each
 *          subsequent slice evaluates the generic body expression using the
 *          current input tile and accumulated output tile.
 */
static LogicalResult tileGenericOp(
    linalg::GenericOp op, linalg::GenericOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter,
    llvm::DenseMap<Value, int64_t> &waitState) {
  auto fallbackToReduce = [&]() -> LogicalResult {
    return rewriteReduceGeneric(op, adaptor, rewriter, waitState);
  };

  Value inCb = adaptor.getInputs().front();
  Value outCb = adaptor.getOutputs().front();
  if (!isa<CBType>(inCb.getType()) || !isa<CBType>(outCb.getType()))
    return fallbackToReduce();

  TileGenericExprAnalysis analysis;
  if (failed(analyzeTileGeneric(op, analysis)))
    return fallbackToReduce();

  auto inType = dyn_cast<ShapedType>(op.getDpsInputs()[0].getType());
  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());

  ArrayRef<int64_t> inShape = inType.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();
  if (inShape[0] <= 0 || inShape[1] != outShape[0] || inShape[2] != outShape[1])
    return fallbackToReduce();

  auto rowTiles = ceilDiv32(outShape[0]);
  auto colTiles = ceilDiv32(outShape[1]);
  if (!rowTiles || !colTiles)
    return fallbackToReduce();

  int64_t outTiles = (*rowTiles) * (*colTiles);
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
  Value inputTilesV = i32(inputTiles);

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
  if (analysis.needsAccumulatorCopy && outCb != inCb)
    rewriter.create<CopyTileInitOp>(loc, outCb);
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
  if (analysis.needsFill)
    rewriter.create<FillTileInitOp>(loc);
  if (analysis.needsPowBinary)
    rewriter.create<PowBinaryTilesInitOp>(loc);
  if (analysis.needsRecip)
    rewriter.create<RecipTileInitOp>(loc);
  if (analysis.needsLog)
    rewriter.create<LogTileInitOp>(loc);

  scf::ForOp outTileLoop =
      scf::ForOp::create(rewriter, loc, zeroI32, outTilesV, oneI32);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(outTileLoop.getBody());
    Value outTileIdx = outTileLoop.getInductionVar();

    TileRegsAcquireOp::create(rewriter, loc);
    rewriter.create<CopyTileOp>(loc, inCb, outTileIdx, zeroI32);
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);
    PackTileOp::create(rewriter, loc, zeroI32, outCb, outTileIdx);
    TileRegsReleaseOp::create(rewriter, loc);

    scf::ForOp reduceLoop =
        scf::ForOp::create(rewriter, loc, oneI32, reduceSlicesV, oneI32);
    {
      OpBuilder::InsertionGuard reduceGuard(rewriter);
      rewriter.setInsertionPointToStart(reduceLoop.getBody());
      Value reduceIdx = reduceLoop.getInductionVar();
      Value sliceOffset = arith::MulIOp::create(rewriter, loc, reduceIdx, outTilesV);
      Value inTileIdx =
          arith::AddIOp::create(rewriter, loc, sliceOffset, outTileIdx);
      TileRegsAcquireOp::create(rewriter, loc);
      if (failed(emitTileGenericExprToReg(analysis.yieldValue, /*dstReg=*/0,
                                          /*tmpRegA=*/1, /*tmpRegB=*/2, op,
                                          adaptor, rewriter, loc, inTileIdx,
                                          outTileIdx,
                                          /*emitInlineInitOps=*/false)))
        return failure();
      TileRegsCommitOp::create(rewriter, loc);
      TileRegsWaitOp::create(rewriter, loc);
      PackTileOp::create(rewriter, loc, zeroI32, outCb, outTileIdx);
      TileRegsReleaseOp::create(rewriter, loc);
    }
    rewriter.setInsertionPointAfter(reduceLoop);
  }
  CBPushBackOp::create(rewriter, loc, outCb, outTilesV);
  waitState[outCb] = 0;
  //CBPopFrontOp::create(rewriter, loc, inCb, inputTilesV);
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
 * @brief Analyze matmul/batch_matmul shapes into a shared tile model.
 *
 * @details For rank-2 inputs this returns the batch-1 specialization; rank-3
 *          inputs are interpreted as true batched matmul. This keeps tile-count
 *          and wait/pack math identical between matmul and batch_matmul.
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

  if (lhsType.getRank() == 2 && rhsType.getRank() == 2 &&
      outType.getRank() == 2) {
    ArrayRef<int64_t> lhs = lhsType.getShape();
    ArrayRef<int64_t> rhs = rhsType.getShape();
    ArrayRef<int64_t> out = outType.getShape();
    if (lhs[1] != rhs[0] || lhs[0] != out[0] || rhs[1] != out[1])
      return std::nullopt;

    auto rt = toI64(getTileDim(lhsType, 0));
    auto kt = toI64(getTileDim(lhsType, 1));
    auto ct = toI64(getTileDim(rhsType, 1));
    auto nt = toI64(getTileDim(outType, 1));
    if (!rt || !kt || !ct || !nt)
      return std::nullopt;

    info.batchSize = 1;
    info.rt = *rt;
    info.kt = *kt;
    info.ct = *ct;
    info.nt = *nt;
  } else if (lhsType.getRank() == 3 && rhsType.getRank() == 3 &&
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

static Value i32Const(ConversionPatternRewriter &rewriter, Location loc,
                      int64_t value) {
  return rewriter.create<arith::ConstantIntOp>(loc, value, 32);
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

struct ReduceRegionAnalysis {
  Value ulX;
  Value ulY;
  Value lrX;
  Value lrY;
  Value participants;
  Value inRegion;
  Value isReducer;
};

struct ReduceRuntimeValues {
  Value inCb;
  Value outCb;
  Value numTiles;
  Value zero;
  Value one;
};

// ReduceCombineOp is defined in ComputeOpToTTKernel.h.

static Value toI32(ConversionPatternRewriter &rewriter, Location loc, Value value) {
  if (!value)
    return {};
  Type type = value.getType();
  if (type.isIndex())
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), value);

  if (type.isInteger(32))
    return value;

  if (auto intTy = dyn_cast<IntegerType>(type)) {
    if (intTy.getWidth() < 32)
      return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI32Type(), value);
    if (intTy.getWidth() > 32)
      return rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), value);
  }

  return {};
}

static LogicalResult validateReducePlacement(::loom::ReduceSumOp op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<scf::IfOp>(parent)) {
      return op.emitOpError("must execute on all participating cores; "
                            "reducer-only guarded control flow is unsupported");
    }
  }
  return success();
}

static FailureOr<ReduceRegionAnalysis>
analyzeReduceRegion(::loom::ReduceSumOp op, ConversionPatternRewriter &rewriter,
                    std::shared_ptr<CompileArgTracker> tracker) {
  if (!tracker)
    return failure();

  Location loc = op.getLoc();
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return failure();

  Value coreX = tracker->getCoreCoordForDim(parentFunc.getOperation(), "x");
  Value coreY = tracker->getCoreCoordForDim(parentFunc.getOperation(), "y");
  if (!coreX || !coreY)
    return failure();

  llvm::DenseMap<Value, Value> i32Cache;
  auto toI32Cached = [&](Value value) -> Value {
    if (!value)
      return {};
    auto it = i32Cache.find(value);
    if (it != i32Cache.end())
      return it->second;
    Value converted = toI32(rewriter, loc, value);
    if (converted)
      i32Cache.try_emplace(value, converted);
    return converted;
  };

  Value ulX = toI32Cached(op.getUlX());
  Value ulY = toI32Cached(op.getUlY());
  Value lrX = toI32Cached(op.getLrX());
  Value lrY = toI32Cached(op.getLrY());
  coreX = toI32(rewriter, loc, coreX);
  coreY = toI32(rewriter, loc, coreY);
  if (!ulX || !ulY || !lrX || !lrY || !coreX || !coreY)
    return failure();

  Value one = i32Const(rewriter, loc, 1);
  auto subI32 = [&](Value lhs, Value rhs) -> Value {
    if (lhs == rhs)
      return i32Const(rewriter, loc, 0);
    return arith::SubIOp::create(rewriter, loc, lhs, rhs);
  };
  auto addI32 = [&](Value lhs, Value rhs) -> Value {
    return arith::AddIOp::create(rewriter, loc, lhs, rhs);
  };

  Value width = addI32(subI32(lrX, ulX), one);
  Value height = addI32(subI32(lrY, ulY), one);
  Value participants = arith::MulIOp::create(rewriter, loc, width, height);
  Value geUlX =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge, coreX, ulX);
  Value leLrX =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sle, coreX, lrX);
  Value geUlY =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge, coreY, ulY);
  Value leLrY =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sle, coreY, lrY);
  Value inRegionX = arith::AndIOp::create(rewriter, loc, geUlX, leLrX);
  Value inRegionY = arith::AndIOp::create(rewriter, loc, geUlY, leLrY);
  Value inRegion = arith::AndIOp::create(rewriter, loc, inRegionX, inRegionY);

  Value isReducerX =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, coreX, ulX);
  Value isReducerY =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, coreY, ulY);
  Value isReducer = arith::AndIOp::create(rewriter, loc, isReducerX, isReducerY);
  Value isReducerInRegion = arith::AndIOp::create(rewriter, loc, inRegion, isReducer);
  return ReduceRegionAnalysis{ulX, ulY, lrX, lrY, participants, inRegion,
                              isReducerInRegion};
}

static FailureOr<ReduceRuntimeValues>
materializeReduceRuntime(::loom::ReduceSumOp op, Value inCb, Value outCb,
                         ConversionPatternRewriter &rewriter,
                         std::shared_ptr<CompileArgTracker> tracker,
                         const ReduceRegionAnalysis &analysis) {
  (void)tracker;
  (void)analysis;

  if (!isa<CBType>(inCb.getType()) || !isa<CBType>(outCb.getType()))
    return failure();

  auto numTilesOpt = getNumTilesFromShapedType(op.getInput().getType());
  if (!numTilesOpt)
    return failure();

  Location loc = op.getLoc();
  Value zero = i32Const(rewriter, loc, 0);
  Value one = i32Const(rewriter, loc, 1);
  Value numTiles = i32Const(rewriter, loc, *numTilesOpt);
  return ReduceRuntimeValues{inCb, outCb, numTiles, zero, one};
}

/**
 * @brief Emit reducer combine op initialization for the selected operation.
 */
static LogicalResult emitReduceCombineInit(ConversionPatternRewriter &rewriter,
                                           Location loc,
                                           ReduceCombineOp combineOp) {
  switch (combineOp) {
  case ReduceCombineOp::Sum:
    rewriter.create<AddBinaryTilesInitOp>(loc);
    return success();
  case ReduceCombineOp::Max:
    return emitError(loc) << "unsupported reduce combine kind: max";
  case ReduceCombineOp::Exp:
    return emitError(loc) << "unsupported reduce combine kind: exp";
  }
  llvm_unreachable("unhandled ReduceCombineOp");
}

/**
 * @brief Emit a single tile combine operation for the selected reducer op.
 */
static LogicalResult emitReduceCombine(ConversionPatternRewriter &rewriter,
                                       Location loc, Value lhsReg,
                                       Value rhsReg, Value dstReg,
                                       ReduceCombineOp combineOp) {
  switch (combineOp) {
  case ReduceCombineOp::Sum:
    rewriter.create<AddBinaryTilesOp>(loc, lhsReg, rhsReg, dstReg);
    return success();
  case ReduceCombineOp::Max:
    return emitError(loc) << "unsupported reduce combine kind: max";
  case ReduceCombineOp::Exp:
    return emitError(loc) << "unsupported reduce combine kind: exp";
  }
  llvm_unreachable("unhandled ReduceCombineOp");
}

static LogicalResult
emitTileAccumulate(ConversionPatternRewriter &rewriter, Location loc,
                   Value dstCb, Value srcCb, Value tiles, Value dstBase,
                   Value srcBase, ReduceCombineOp combineOp) {
  Value zero = i32Const(rewriter, loc, 0);
  Value one = i32Const(rewriter, loc, 1);
  rewriter.create<CopyTileInitOp>(loc, dstCb);
  if (srcCb != dstCb)
    rewriter.create<CopyTileInitOp>(loc, srcCb);
  if (failed(emitReduceCombineInit(rewriter, loc, combineOp)))
    return failure();
  rewriter.create<CBWaitFrontOp>(loc, srcCb, tiles);
  scf::ForOp tileLoop = scf::ForOp::create(rewriter, loc, zero, tiles, one);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(tileLoop.getBody());
    Value tileIdx = tileLoop.getInductionVar();
    Value dstIdx = arith::AddIOp::create(rewriter, loc, dstBase, tileIdx);
    Value srcIdx = arith::AddIOp::create(rewriter, loc, srcBase, tileIdx);
    TileRegsAcquireOp::create(rewriter, loc);
    rewriter.create<CopyTileOp>(loc, dstCb, dstIdx, zero);
    rewriter.create<CopyTileOp>(loc, srcCb, srcIdx, one);
    if (failed(emitReduceCombine(rewriter, loc, zero, one, zero, combineOp)))
      return failure();
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);
    PackTileOp::create(rewriter, loc, zero, dstCb, dstIdx);
    TileRegsReleaseOp::create(rewriter, loc);
  }
  rewriter.create<CBPopFrontOp>(loc, srcCb, tiles);
  return success();
}

static void copyTile(ConversionPatternRewriter &rewriter,
  Location loc, Value inputCb, Value outputCb, Value tiles) {
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
  //TODO: tmp use for split_k first, CBPopFront should only be controlled by semaphore.give
  CBPopFrontOp::create(rewriter, loc, inputCb, tiles);
}

static void emitWorkerPreparePayload(ConversionPatternRewriter &rewriter,
                                     Location loc,
                                     const ReduceRuntimeValues &runtime) {
  Value zero = runtime.zero;
  Value one = runtime.one;
  Value reg0 = i32Const(rewriter, loc, 0);
  CBWaitFrontOp::create(rewriter, loc, runtime.inCb, runtime.numTiles);
  CBReserveBackOp::create(rewriter, loc, runtime.outCb, runtime.numTiles);
  rewriter.create<CopyTileInitOp>(loc, runtime.inCb);
  scf::ForOp tileLoop =
      scf::ForOp::create(rewriter, loc, zero, runtime.numTiles, one);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(tileLoop.getBody());
    Value tileIdx = tileLoop.getInductionVar();
    TileRegsAcquireOp::create(rewriter, loc);
    rewriter.create<CopyTileOp>(loc, runtime.inCb, tileIdx, reg0);
    TileRegsCommitOp::create(rewriter, loc);
    TileRegsWaitOp::create(rewriter, loc);
    PackTileOp::create(rewriter, loc, reg0, runtime.outCb, tileIdx);
    TileRegsReleaseOp::create(rewriter, loc);
  }
  CBPushBackOp::create(rewriter, loc, runtime.outCb, runtime.numTiles);
}

static LogicalResult
emitReducerGather(ConversionPatternRewriter &rewriter, Location loc,
                  const ReduceRegionAnalysis &analysis,
                  const ReduceRuntimeValues &runtime, ReduceProtocol protocol,
                  ReduceCombineOp combineOp) {
  bool singleSlot = protocol == ReduceProtocol::SingleSlot;
  //copy reducer's result to output cb
  copyTile(rewriter, loc, runtime.inCb, runtime.outCb, runtime.numTiles);
  scf::ForOp workerLoop = scf::ForOp::create(
      rewriter, loc, runtime.one, analysis.participants, runtime.one);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(workerLoop.getBody());
    Value rank = workerLoop.getInductionVar();
    Value srcCb = singleSlot ? runtime.inCb : runtime.outCb;
    Value srcBase =
        singleSlot ? runtime.zero
                   : arith::MulIOp::create(rewriter, loc, rank, runtime.numTiles);
    if (failed(emitTileAccumulate(rewriter, loc, runtime.outCb, srcCb,
                                  runtime.numTiles, runtime.zero, srcBase,
                                  combineOp)))
      return failure();
  }

  CBPushBackOp::create(rewriter, loc, runtime.outCb, runtime.numTiles);
  return success();
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

/**
 * @brief Collect init-op requirements for tile-generic body expressions.
 */
static LogicalResult analyzeTileGenericExpr(Value exprValue, linalg::GenericOp op,
                                            TileGenericExprAnalysis &analysis) {
  auto handleLeaf = [&](Value value) -> bool {
    auto operand = getTileGenericBodyOperand(op, value);
    if (!operand)
      return false;
    if (*operand == TileGenericBodyOperand::Accumulator)
      analysis.needsAccumulatorCopy = true;
    return true;
  };
  auto handleConst = [&]() { analysis.needsFill = true; };
  auto handlePowBaseConst = [&]() { analysis.needsFill = true; };
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
    case GenericExprFeature::Fill:
      analysis.needsFill = true;
      return;
    }
  };
  return analyzeGenericExprTree(
      exprValue, op, handleLeaf, handleConst, handlePowBaseConst, setFlag,
      /*allowMaximumFOp=*/true);
}

/**
 * @brief Validate tile generic shape and record body-expression requirements.
 */
static LogicalResult analyzeTileGeneric(linalg::GenericOp op,
                                        TileGenericExprAnalysis &analysis) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
    return failure();

  auto yieldOp = dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return failure();

  analysis.yieldValue = yieldOp.getValues().front();
  return analyzeTileGenericExpr(analysis.yieldValue, op, analysis);
}

/**
 * @brief Shared recursive emission for generic expression trees.
 *
 * @details This helper emits all non-leaf expression operators shared by
 *          FlashAttention elementwise and tile-generic lowering. Callers
 *          provide leaf materialization and operand-order policy.
 */
template <typename LeafEmitterT, typename RhsFirstPolicyT>
static LogicalResult emitGenericExprToRegImpl(
    Value exprValue, int dstReg, int tmpRegA, int tmpRegB, linalg::GenericOp op,
    linalg::GenericOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
    Location loc, bool emitInlineInitOps, bool allowMaximumFOp,
    LeafEmitterT &&emitLeafToReg, RhsFirstPolicyT &&emitRhsFirstForOperands) {
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

/**
 * @brief Emit tile-generic body expressions using TTKernel tile ops.
 */
static LogicalResult emitTileGenericExprToReg(
    Value exprValue, int dstReg, int tmpRegA, int tmpRegB, linalg::GenericOp op,
    linalg::GenericOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
    Location loc, Value inputTileIdx, Value accumulatorTileIdx,
    bool emitInlineInitOps) {
  auto emitLeaf = [&](Value value, int reg, bool &handled) -> LogicalResult {
    auto operand = getTileGenericBodyOperand(op, value);
    if (!operand) {
      handled = false;
      return success();
    }

    handled = true;
    Value regVal = i32Const(rewriter, loc, reg);
    Value sourceCb = *operand == TileGenericBodyOperand::Input
                         ? adaptor.getInputs()[0]
                         : adaptor.getOutputs()[0];
    Value sourceTileIdx = *operand == TileGenericBodyOperand::Input
                              ? inputTileIdx
                              : accumulatorTileIdx;
    if (emitInlineInitOps)
      rewriter.create<CopyTileInitOp>(loc, sourceCb);
    rewriter.create<CopyTileOp>(loc, sourceCb, sourceTileIdx, regVal);
    return success();
  };

  auto emitRhsFirstForOperands = [&](Value lhs, Value rhs) -> bool {
    (void)lhs;
    (void)rhs;
    return false;
  };

  return emitGenericExprToRegImpl(
      exprValue, dstReg, tmpRegA, tmpRegB, op, adaptor, rewriter, loc,
      emitInlineInitOps, /*allowMaximumFOp=*/true, emitLeaf,
      emitRhsFirstForOperands);
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

  SmallVector<int64_t, 4> outTileShape;
  outTileShape.reserve(rank);
  for (int64_t dimSize : outType.getShape()) {
    auto dimTiles = ceilDiv32(dimSize);
    if (!dimTiles)
      return failure();
    outTileShape.push_back(*dimTiles);
  }

  bool allIdentity = hasAllIdentityMapsForRank(op, rank);
  std::optional<ElementwiseBroadcastInfo> broadcast;
  if (!allIdentity) {
    unsigned numInputs = op.getNumDpsInputs();
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != numInputs + 1)
      return failure();
    if (!isIdentityMapForRank(maps[numInputs], rank))
      return failure();

    for (unsigned i = 0; i < numInputs; ++i) {
      if (isIdentityMapForRank(maps[i], rank))
        continue;

      auto droppedDim = getDroppedDimFromBroadcastMap(maps[i], rank);
      if (!droppedDim || broadcast)
        return failure();

      auto suffixTiles = getSuffixTilesAfterDim(outTileShape, *droppedDim);
      if (!suffixTiles || *suffixTiles <= 0)
        return failure();

      int64_t droppedDimTiles = outTileShape[*droppedDim];
      if (droppedDimTiles <= 0)
        return failure();

      ElementwiseBroadcastInfo info;
      info.inputIdx = i;
      info.droppedDim = *droppedDim;
      info.kind = classifyElementwiseBroadcastKind(*droppedDim, rank);
      info.droppedDimTiles = droppedDimTiles;
      info.suffixTiles = *suffixTiles;
      broadcast = info;
    }

    // Non-identity elementwise maps must be one supported broadcast input.
    if (!broadcast)
      return failure();
  }
  analysis.broadcast = broadcast;

  auto outTiles = getNumTilesFromShapedType(op.getDpsInits()[0].getType());
  if (!outTiles)
    return failure();
  analysis.outTiles = *outTiles;

  analysis.inputWaitTiles.assign(op.getNumDpsInputs(), *outTiles);
  analysis.usedInputs.resize(op.getNumDpsInputs());
  analysis.usedInputs.reset();

  if (analysis.broadcast) {
    auto inputTiles = getNumTilesFromShapedType(
        op.getDpsInputs()[analysis.broadcast->inputIdx].getType());
    if (!inputTiles || *inputTiles <= 0)
      return failure();
    analysis.inputWaitTiles[analysis.broadcast->inputIdx] = *inputTiles;
  }

  auto yieldOp = dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return failure();
  analysis.yieldValue = yieldOp.getValues().front();

  if (failed(analyzeElementwiseExpr(analysis.yieldValue, op, analysis)))
    return failure();

  if (analysis.broadcast &&
      analysis.usedInputs.test(analysis.broadcast->inputIdx) &&
      analysis.broadcast->kind != ElementwiseBroadcastKind::BatchBroadcast)
    analysis.needsUnaryBcast = true;

  return success();
}

static bool exprContainsBodyInput(Value exprValue, linalg::GenericOp op,
                                  unsigned inputIdx) {
  if (auto idx = getBodyInputIndex(op, exprValue))
    return *idx == inputIdx;

  if (getConstFloatValue(exprValue).has_value())
    return false;

  Operation *defOp = exprValue.getDefiningOp();
  if (!defOp || defOp->getBlock() != &op.getRegion().front())
    return false;

  if (auto mulOp = dyn_cast<arith::MulFOp>(defOp))
    return exprContainsBodyInput(mulOp.getLhs(), op, inputIdx) ||
           exprContainsBodyInput(mulOp.getRhs(), op, inputIdx);

  if (auto addOp = dyn_cast<arith::AddFOp>(defOp))
    return exprContainsBodyInput(addOp.getLhs(), op, inputIdx) ||
           exprContainsBodyInput(addOp.getRhs(), op, inputIdx);

  if (auto subOp = dyn_cast<arith::SubFOp>(defOp))
    return exprContainsBodyInput(subOp.getLhs(), op, inputIdx) ||
           exprContainsBodyInput(subOp.getRhs(), op, inputIdx);

  if (auto divOp = dyn_cast<arith::DivFOp>(defOp))
    return exprContainsBodyInput(divOp.getLhs(), op, inputIdx) ||
           exprContainsBodyInput(divOp.getRhs(), op, inputIdx);

  if (auto powOp = dyn_cast<math::PowFOp>(defOp))
    return exprContainsBodyInput(powOp.getLhs(), op, inputIdx) ||
           exprContainsBodyInput(powOp.getRhs(), op, inputIdx);

  if (auto logOp = dyn_cast<math::LogOp>(defOp))
    return exprContainsBodyInput(logOp.getOperand(), op, inputIdx);

  if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
    Value lhs;
    Value rhs;
    if (!matchMaxSelect(selectOp, lhs, rhs))
      return false;
    return exprContainsBodyInput(lhs, op, inputIdx) ||
           exprContainsBodyInput(rhs, op, inputIdx);
  }

  return false;
}

static bool shouldEmitRhsFirstForBroadcast(
    Value lhs, Value rhs, linalg::GenericOp op,
    std::optional<unsigned> broadcastInput, bool emitInlineInitOps) {
  if (!emitInlineInitOps || !broadcastInput)
    return false;
  bool lhsUsesRowBcast = exprContainsBodyInput(lhs, op, *broadcastInput);
  bool rhsUsesRowBcast = exprContainsBodyInput(rhs, op, *broadcastInput);
  return rhsUsesRowBcast && !lhsUsesRowBcast;
}

static LogicalResult emitElementwiseExprToReg(
    Value exprValue, int dstReg, int tmpRegA, int tmpRegB, linalg::GenericOp op,
    linalg::GenericOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
    Location loc, Value tileIdx, std::optional<Value> broadcastTileIdx,
    std::optional<unsigned> broadcastInput, std::optional<BcastType> bcastType,
    bool emitInlineInitOps, std::optional<unsigned> broadcastRefInput) {
  auto emitLeaf = [&](Value value, int reg, bool &handled) -> LogicalResult {
    auto inputIdx = getBodyInputIndex(op, value);
    if (!inputIdx) {
      handled = false;
      return success();
    }

    handled = true;
    Value regVal = i32Const(rewriter, loc, reg);
    if (broadcastInput && *inputIdx == *broadcastInput) {
      if (!broadcastTileIdx)
        return failure();

      if (bcastType) {
        if (emitInlineInitOps) {
          if (!broadcastRefInput)
            return failure();
          rewriter.create<UnaryBcastInitOp>(
              loc, adaptor.getInputs()[*broadcastInput],
              adaptor.getInputs()[*broadcastRefInput], *bcastType);
        }
        rewriter.create<UnaryBcastTileOp>(loc, adaptor.getInputs()[*inputIdx],
                                          *broadcastTileIdx, regVal, *bcastType);
      } else {
        if (emitInlineInitOps)
          rewriter.create<CopyTileInitOp>(loc, adaptor.getInputs()[*inputIdx]);
        rewriter.create<CopyTileOp>(loc, adaptor.getInputs()[*inputIdx],
                                    *broadcastTileIdx, regVal);
      }
      return success();
    }

    if (emitInlineInitOps)
      rewriter.create<CopyTileInitOp>(loc, adaptor.getInputs()[*inputIdx]);
    rewriter.create<CopyTileOp>(loc, adaptor.getInputs()[*inputIdx], tileIdx,
                                regVal);
    return success();
  };

  auto emitRhsFirstForOperands = [&](Value lhs, Value rhs) -> bool {
    return shouldEmitRhsFirstForBroadcast(lhs, rhs, op, broadcastInput,
                                         emitInlineInitOps);
  };

  return emitGenericExprToRegImpl(
      exprValue, dstReg, tmpRegA, tmpRegB, op, adaptor, rewriter, loc,
      emitInlineInitOps, /*allowMaximumFOp=*/false, emitLeaf,
      emitRhsFirstForOperands);
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
  if (!inType || !outType || inType.getRank() != 3 || outType.getRank() != 2)
    return failure();

  auto bTiles = getTileDim(inType, 0);
  auto mTiles = getTileDim(inType, 1);
  auto nTiles = getTileDim(inType, 2);
  auto outTiles = getNumTilesFromShapedType(outType);
  if (!bTiles || !mTiles || !nTiles || !outTiles)
    return failure();

  int64_t rows = (*bTiles) * (*mTiles);
  int64_t cols = *nTiles;
  int64_t numTiles = rows * cols;

  Location loc = op.getLoc();
  Value outTilesV = i32Const(rewriter, loc, rows);
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
                                               llvm::DenseMap<Value, int64_t> &waitState) {
  if (adaptor.getOutputs().size() != 1 || adaptor.getInputs().empty())
    return failure();

  ElementwiseAnalysis analysis;
  if (failed(analyzeElementwiseGeneric(op, analysis)))
    return failure();

  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(outCb.getType()))
    return failure();
  for (Value inputCb : adaptor.getInputs())
    if (!isa<CBType>(inputCb.getType()))
      return failure();

  Location loc = op.getLoc();
  bool outAliasesInput = hasOutputAlias(outCb, adaptor.getInputs());

  for (auto [idx, inputCb] : llvm::enumerate(adaptor.getInputs())) {
    if (!analysis.usedInputs.test(idx))
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
  for (auto [idx, inputCb] : llvm::enumerate(adaptor.getInputs())) {
    if (analysis.usedInputs.test(idx)) {
      inCbForInit = inputCb;
      break;
    }
  }
  rewriter.create<InitSFPUOp>(loc, inCbForInit, outCb);
  bool emitInlineInitOps = analysis.needsUnaryBcast;
  std::optional<unsigned> broadcastInput;
  std::optional<unsigned> broadcastRefInput;
  std::optional<BcastType> unaryBcastType;
  if (analysis.broadcast &&
      analysis.usedInputs.test(analysis.broadcast->inputIdx)) {
    broadcastInput = analysis.broadcast->inputIdx;
    if (analysis.needsUnaryBcast) {
      unaryBcastType = getUnaryBcastType(analysis.broadcast->kind);
      if (!unaryBcastType)
        return failure();

      unsigned refInput = *broadcastInput;
      for (unsigned i = 0; i < adaptor.getInputs().size(); ++i) {
        if (i != *broadcastInput && analysis.usedInputs.test(i)) {
          refInput = i;
          break;
        }
      }
      broadcastRefInput = refInput;
    }
  }

  if (!emitInlineInitOps) {
    for (auto [idx, inputCb] : llvm::enumerate(adaptor.getInputs())) {
      if (!analysis.usedInputs.test(idx))
        continue;
      if (broadcastInput && analysis.needsUnaryBcast &&
          idx == *broadcastInput)
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
  }

  LogicalResult result = emitElementwiseTiles(
      rewriter, loc, outCb, analysis.outTiles, [&](Value tileIdx) -> LogicalResult {
        std::optional<Value> broadcastTileIdx;
        if (broadcastInput && analysis.broadcast) {
          const ElementwiseBroadcastInfo &broadcastInfo = *analysis.broadcast;
          Value suffixTiles =
              i32Const(rewriter, loc, broadcastInfo.suffixTiles);
          Value droppedDimSpan = i32Const(
              rewriter, loc,
              broadcastInfo.droppedDimTiles * broadcastInfo.suffixTiles);
          Value prefix = rewriter.create<arith::DivUIOp>(loc, tileIdx,
                                                         droppedDimSpan);
          if (broadcastInfo.suffixTiles == 1) {
            broadcastTileIdx = prefix;
          } else {
            Value suffixOffset =
                rewriter.create<arith::RemUIOp>(loc, tileIdx, suffixTiles);
            Value prefixBase =
                rewriter.create<arith::MulIOp>(loc, prefix, suffixTiles);
            broadcastTileIdx =
                rewriter.create<arith::AddIOp>(loc, prefixBase, suffixOffset);
          }
        }
        return emitElementwiseExprToReg(
            analysis.yieldValue, /*dstReg=*/0, /*tmpRegA=*/1, /*tmpRegB=*/2, op,
            adaptor, rewriter, loc, tileIdx, broadcastTileIdx, broadcastInput,
            unaryBcastType, emitInlineInitOps, broadcastRefInput);
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

class ConvertLinalgMatmulOp : public OpConversionPattern<linalg::MatmulOp> {
public:
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

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
    Value outBuffer = op.getOutputs()[0];

    // Ensure operands are TTKernel CBs.
    if (!isa<CBType>(in0Cb.getType()) || !isa<CBType>(in1Cb.getType()) ||
        !isa<CBType>(outCb.getType()))
      return failure();

    auto lhsShapedType = dyn_cast<ShapedType>(op.getInputs()[0].getType());
    auto rhsShapedType = dyn_cast<ShapedType>(op.getInputs()[1].getType());
    auto outShapedType = dyn_cast<ShapedType>(op.getOutputs()[0].getType());
    std::optional<MatmulTileInfo> tileInfo =
        getMatmulTileInfo(lhsShapedType, rhsShapedType, outShapedType);
    if (!tileInfo)
      return failure();

    // Matmul lowering is the batch=1 specialization of batch_matmul lowering.
    if (tileInfo->batchSize != 1)
      return failure();

    Value zeroI32;
    Value in0TileIdx;
    Value in1TileIdx;
    Value dstTileIdx;
    Value transpose;
    Value ctDim;
    Value rtDim;
    Value ntDim;
    Value ktDim;
    //TODO: move matmul_block_init just before the code, only use for flashattn
    zeroI32 = rewriter.create<arith::ConstantIntOp>(
      loc, 0, 32);
    in0TileIdx = zeroI32;
    in1TileIdx = zeroI32;
    dstTileIdx = zeroI32;
    transpose = zeroI32;
    ctDim = i32Const(rewriter, loc, tileInfo->ct);
    rtDim = i32Const(rewriter, loc, tileInfo->rt);
    ntDim = i32Const(rewriter, loc, tileInfo->nt);
    ktDim = i32Const(rewriter, loc, tileInfo->kt);

    rewriter.create<MatmulBlockInitShortOp>(
        loc, TypeRange{},
        ValueRange{in0Cb, in1Cb, transpose, ctDim, rtDim, ktDim});


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

/*     {
      OpBuilder::InsertionGuard guard(rewriter);
      bool placeInitAtKernelStart = false;

      if (auto parentFunc = op->getParentOfType<func::FuncOp>()) {
        Block &entry = parentFunc.front();
        Operation *lastGetArgValOp = nullptr;
        for (Operation &entryOp : entry)
          if (isa<GetArgValOp>(entryOp))
            lastGetArgValOp = &entryOp;

        auto isAvailableAtEntry = [&](Value value) -> bool {
          if (auto blockArg = dyn_cast<BlockArgument>(value))
            return blockArg.getOwner() == &entry;

          Operation *defOp = value.getDefiningOp();
          if (!defOp || defOp->getBlock() != &entry)
            return false;

          if (!lastGetArgValOp)
            return true;

          return defOp == lastGetArgValOp ||
                 defOp->isBeforeInBlock(lastGetArgValOp);
        };

        placeInitAtKernelStart = isAvailableAtEntry(in0Cb) &&
                                 isAvailableAtEntry(in1Cb) &&
                                 isAvailableAtEntry(outCb);

        if (placeInitAtKernelStart) {
          if (lastGetArgValOp)
            rewriter.setInsertionPointAfter(lastGetArgValOp);
          else
            rewriter.setInsertionPointToStart(&entry);
        }
      }

      zeroI32 = rewriter.create<arith::ConstantIntOp>(
          loc, 0, 32);
      in0TileIdx = zeroI32;
      in1TileIdx = zeroI32;
      dstTileIdx = zeroI32;
      transpose = zeroI32;
      ctDim = rewriter.create<arith::ConstantIntOp>(loc, ctVal, 32);
      rtDim = rewriter.create<arith::ConstantIntOp>(loc, rtVal, 32);
      ntDim = rewriter.create<arith::ConstantIntOp>(loc, ntVal, 32);
      ktDim = rewriter.create<arith::ConstantIntOp>(loc, ktVal, 32);

      rewriter.create<MatmulBlockInitOp>(
          loc, TypeRange{},
          ValueRange{in0Cb, in1Cb, outCb, transpose, ctDim, rtDim, ktDim});
    } */

    Operation *materializationScopeOp =
        findMatmulOutputMaterializationScope(outBuffer, op.getOperation());

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(materializationScopeOp);
      TileRegsAcquireOp::create(rewriter, loc);
    }

    rewriter.create<ExperimentalMatmulBlockOp>(
        loc, TypeRange{},
        ValueRange{in0Cb, in1Cb, in0TileIdx, in1TileIdx, dstTileIdx, transpose,
                   ctDim, rtDim, ktDim, ntDim});

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(materializationScopeOp);

      Value outCbNumTiles = i32Const(rewriter, loc, tileInfo->outTilesTotal);
      Value lowerBound = i32Const(rewriter, loc, 0);
      Value step = i32Const(rewriter, loc, 1);

      // Materialize tile-register results into L1 CB at the chosen scope.
      CBReserveBackOp::create(rewriter, loc, outCb, outCbNumTiles);
      TileRegsCommitOp::create(rewriter, loc);
      TileRegsWaitOp::create(rewriter, loc);

      scf::ForOp packLoop =
          scf::ForOp::create(rewriter, loc, lowerBound, outCbNumTiles, step);
      rewriter.setInsertionPointToStart(packLoop.getBody());
      Value i = packLoop.getInductionVar();
      PackTileOp::create(rewriter, loc, i, outCb, i);
      rewriter.setInsertionPointAfter(packLoop);
      TileRegsReleaseOp::create(rewriter, loc);
      CBPushBackOp::create(rewriter, loc, outCb, outCbNumTiles);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

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
    Value zero = i32Const(rewriter, loc, 0);
    Value tileCount = i32Const(rewriter, loc, *inTiles);

    CBPopFrontOp::create(rewriter, loc, outCb, tileCount);
    CBWaitFrontOp::create(rewriter, loc, inCb, tileCount);
    CBReserveBackOp::create(rewriter, loc, outCb, tileCount);
    rewriter.create<CopyTileInitOp>(loc, inCb);
    
    //TODO: maybe need to redesign the tileIdx of input/output cb and DST here
    for (int64_t i = 0; i < *inTiles; ++i) {
      Value tileIdx = i32Const(rewriter, loc, i);
      TileRegsAcquireOp::create(rewriter, loc);
      rewriter.create<CopyTileOp>(loc, inCb, tileIdx, zero);
      TileRegsCommitOp::create(rewriter, loc);
      TileRegsWaitOp::create(rewriter, loc);
      PackTileOp::create(rewriter, loc, zero, outCb, tileIdx);
      TileRegsReleaseOp::create(rewriter, loc);
    }

    CBPushBackOp::create(rewriter, loc, outCb, tileCount);
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Map a source reduce op to its combine kind.
 *
 * @details Today only `loom.reduce_sum` exists, mapping to `Sum`. When the
 *          dialect gains a generic `loom.reduce` with a `combine_kind`
 *          attribute, this function should read that attribute instead.
 */
static ReduceCombineOp getReduceCombineOp(Operation *op) {
  if (isa<::loom::ReduceSumOp>(op))
    return ReduceCombineOp::Sum;
  llvm_unreachable("unknown reduce op kind");
}

class ConvertLoomReduceComputeOp
    : public OpConversionPattern<::loom::ReduceSumOp> {
public:
  ConvertLoomReduceComputeOp(TypeConverter &typeConverter, MLIRContext *context,
                             std::shared_ptr<CompileArgTracker> tracker,
                             ReduceProtocol protocol)
      : OpConversionPattern<::loom::ReduceSumOp>(typeConverter, context),
        tracker(std::move(tracker)), protocol(protocol) {}

  LogicalResult
  matchAndRewrite(::loom::ReduceSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isComputeKernel(op.getOperation()))
      return failure();
    if (failed(validateReducePlacement(op)))
      return failure();
    if (adaptor.getOperands().size() < 2)
      return failure();

    auto analysis = analyzeReduceRegion(op, rewriter, tracker);
    if (failed(analysis)) {
      return rewriter.notifyMatchFailure(op, "failed to analyze reduce region");
    }

    Value inCb = adaptor.getOperands().front();
    Value outCb = adaptor.getOperands()[1];
    auto runtime =
        materializeReduceRuntime(op, inCb, outCb, rewriter, tracker, *analysis);
    if (failed(runtime)) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize reduce runtime arguments");
    }

    ReduceCombineOp combineOp = getReduceCombineOp(op);
    Location loc = op.getLoc();
    scf::IfOp inRegionIf =
        rewriter.create<scf::IfOp>(loc, analysis->inRegion,
                                   /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&inRegionIf.getThenRegion().front());
      scf::IfOp roleIf =
          rewriter.create<scf::IfOp>(loc, analysis->isReducer,
                                     /*withElseRegion=*/true);

      rewriter.setInsertionPointToStart(&roleIf.getThenRegion().front());
      if (failed(emitReducerGather(rewriter, loc, *analysis, *runtime, protocol,
                                   combineOp)))
        return failure();

      rewriter.setInsertionPointToStart(&roleIf.getElseRegion().front());
      emitWorkerPreparePayload(rewriter, loc, *runtime);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  std::shared_ptr<CompileArgTracker> tracker;
  ReduceProtocol protocol;
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

class ConvertFlashAttentionGenericOp
    : public OpConversionPattern<linalg::GenericOp> {
public:
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

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
      return rewriteElementwiseGeneric(op, adaptor, rewriter, waitState);
    }

    return failure();
  }

private:
  mutable llvm::DenseMap<Value, int64_t> waitState;
  mutable func::FuncOp activeFunc;
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

void mlir::loom::populateComputeOpConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context, std::shared_ptr<CompileArgTracker> tracker,
    ReduceProtocol reduceProtocol) {
  (void)tracker;
  (void)reduceProtocol;
  patterns.add<ConvertLinalgFillOp>(typeConverter, context);
  patterns.add<ConvertLinalgCopyOp>(typeConverter, context);
  patterns.add<ConvertLinalgMatmulOp>(typeConverter, context);
  patterns.add<ConvertLinalgBatchMatmulOp>(typeConverter, context);
  patterns.add<ConvertMemrefCollapseShapeOp>(typeConverter, context);
  patterns.add<ConvertFlashAttentionGenericOp>(typeConverter, context);
}
