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

using namespace mlir;
using namespace tt::ttkernel;

namespace {

enum class FlashAttentionGenericKind {
  Reduction,
  Elementwise
};

struct ElementwiseAnalysis {
  Value yieldValue;
  int64_t outTiles = 0;
  std::optional<unsigned> rowBcastInput;
  std::optional<int64_t> colTiles;
  SmallVector<int64_t, 4> inputWaitTiles;
  llvm::SmallBitVector usedInputs;

  bool needsBinopWithScalar = false;
  bool needsUnaryBcast = false;
  bool needsSubBinary = false;
  bool needsAddBinary = false;
  bool needsMulBinary = false;
  bool needsBinaryMax = false;
  bool needsExp = false;
  bool needsRecip = false;
};

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

static bool isUsedAsLinalgOutput(Operation *user, Value value) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(user);
  if (!linalgOp)
    return false;
  return llvm::is_contained(linalgOp.getDpsInits(), value);
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

static bool isBatchRowMap3D(AffineMap map) {
  if (!map || map.getNumDims() != 3 || map.getNumSymbols() != 0 ||
      map.getNumResults() != 2)
    return false;
  auto d0 = dyn_cast<AffineDimExpr>(map.getResult(0));
  auto d1 = dyn_cast<AffineDimExpr>(map.getResult(1));
  return d0 && d1 && d0.getPosition() == 0 && d1.getPosition() == 1;
}

static bool hasAllIdentityMapsForRank(linalg::GenericOp op, unsigned rank) {
  if (op.getNumLoops() != rank)
    return false;
  for (AffineMap map : op.getIndexingMapsArray())
    if (!isIdentityMapForRank(map, rank))
      return false;
  return true;
}

static std::optional<unsigned> getRowBroadcastInputIndex3D(linalg::GenericOp op) {
  if (op.getNumLoops() != 3 || op.getNumDpsInits() != 1)
    return std::nullopt;

  unsigned numInputs = op.getNumDpsInputs();
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != numInputs + 1)
    return std::nullopt;
  if (!isIdentityMapForRank(maps[numInputs], 3))
    return std::nullopt;

  std::optional<unsigned> rowInputIndex;
  for (unsigned i = 0; i < numInputs; ++i) {
    if (isBatchRowMap3D(maps[i])) {
      if (rowInputIndex)
        return std::nullopt;
      rowInputIndex = i;
      continue;
    }
    if (!isIdentityMapForRank(maps[i], 3))
      return std::nullopt;
  }

  return rowInputIndex;
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


static std::optional<int64_t> ceilDiv32(int64_t value) {
  if (value <= 0)
    return std::nullopt;
  return (value + 31) / 32;
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

template <typename BuilderFn>
static LogicalResult emitElementwiseTiles(
    ConversionPatternRewriter &rewriter, Location loc, Value outCb, int64_t outTiles,
    BuilderFn &&builder) {
  for (int64_t i = 0; i < outTiles; ++i) {
    TileRegsAcquireOp::create(rewriter, loc);
    if (failed(builder(i)))
      return failure();
    PackTileOp::create(rewriter, loc, i32Const(rewriter, loc, 0), outCb,
                       i32Const(rewriter, loc, i));
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

static LogicalResult analyzeElementwiseExpr(Value exprValue, linalg::GenericOp op,
                                            ElementwiseAnalysis &analysis) {
  if (auto inputIdx = getBodyInputIndex(op, exprValue)) {
    analysis.usedInputs.set(*inputIdx);
    return success();
  }

  if (getConstFloatValue(exprValue).has_value())
    return success();

  Operation *defOp = exprValue.getDefiningOp();
  if (!defOp || defOp->getBlock() != &op.getRegion().front())
    return failure();

  if (auto mulOp = dyn_cast<arith::MulFOp>(defOp)) {
    auto lhsConst = getConstFloatValue(mulOp.getLhs());
    auto rhsConst = getConstFloatValue(mulOp.getRhs());
    if (lhsConst && !rhsConst) {
      analysis.needsBinopWithScalar = true;
      return analyzeElementwiseExpr(mulOp.getRhs(), op, analysis);
    }
    if (rhsConst && !lhsConst) {
      analysis.needsBinopWithScalar = true;
      return analyzeElementwiseExpr(mulOp.getLhs(), op, analysis);
    }
    analysis.needsMulBinary = true;
    if (failed(analyzeElementwiseExpr(mulOp.getLhs(), op, analysis)))
      return failure();
    return analyzeElementwiseExpr(mulOp.getRhs(), op, analysis);
  }

  if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
    analysis.needsAddBinary = true;
    if (failed(analyzeElementwiseExpr(addOp.getLhs(), op, analysis)))
      return failure();
    return analyzeElementwiseExpr(addOp.getRhs(), op, analysis);
  }

  if (auto subOp = dyn_cast<arith::SubFOp>(defOp)) {
    analysis.needsSubBinary = true;
    if (failed(analyzeElementwiseExpr(subOp.getLhs(), op, analysis)))
      return failure();
    return analyzeElementwiseExpr(subOp.getRhs(), op, analysis);
  }

  if (auto divOp = dyn_cast<arith::DivFOp>(defOp)) {
    analysis.needsRecip = true;
    analysis.needsMulBinary = true;
    if (failed(analyzeElementwiseExpr(divOp.getLhs(), op, analysis)))
      return failure();
    return analyzeElementwiseExpr(divOp.getRhs(), op, analysis);
  }

  if (auto powOp = dyn_cast<math::PowFOp>(defOp)) {
    if (!getConstFloatValue(powOp.getLhs()).has_value())
      return failure();
    analysis.needsExp = true;
    return analyzeElementwiseExpr(powOp.getRhs(), op, analysis);
  }

  if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
    Value lhs;
    Value rhs;
    if (!matchMaxSelect(selectOp, lhs, rhs))
      return failure();
    analysis.needsBinaryMax = true;
    if (failed(analyzeElementwiseExpr(lhs, op, analysis)))
      return failure();
    return analyzeElementwiseExpr(rhs, op, analysis);
  }

  return failure();
}

static LogicalResult analyzeElementwiseGeneric(linalg::GenericOp op,
                                               ElementwiseAnalysis &analysis) {
  if (op.getNumDpsInits() != 1)
    return failure();

  for (utils::IteratorType iterType : op.getIteratorTypesArray())
    if (iterType == utils::IteratorType::reduction)
      return failure();

  unsigned rank = op.getNumLoops();
  bool allIdentity = hasAllIdentityMapsForRank(op, rank);
  std::optional<unsigned> rowBcastInput;
  if (!allIdentity) {
    if (rank != 3)
      return failure();
    rowBcastInput = getRowBroadcastInputIndex3D(op);
    if (!rowBcastInput)
      return failure();
  }
  analysis.rowBcastInput = rowBcastInput;

  auto outTiles = getNumTilesFromShapedType(op.getDpsInits()[0].getType());
  if (!outTiles)
    return failure();
  analysis.outTiles = *outTiles;

  analysis.inputWaitTiles.assign(op.getNumDpsInputs(), *outTiles);
  analysis.usedInputs.resize(op.getNumDpsInputs());
  analysis.usedInputs.reset();

  if (analysis.rowBcastInput) {
    auto rowTiles =
        getNumTilesFromShapedType(op.getDpsInputs()[*analysis.rowBcastInput].getType());
    auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
    auto colTiles = outType ? getTileDim(outType, 2) : std::nullopt;
    if (!rowTiles || !colTiles || *colTiles <= 0)
      return failure();
    analysis.inputWaitTiles[*analysis.rowBcastInput] = *rowTiles;
    analysis.colTiles = *colTiles;
  }

  auto yieldOp = dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return failure();
  analysis.yieldValue = yieldOp.getValues().front();

  if (failed(analyzeElementwiseExpr(analysis.yieldValue, op, analysis)))
    return failure();

  if (analysis.rowBcastInput && analysis.usedInputs.test(*analysis.rowBcastInput))
    analysis.needsUnaryBcast = true;

  return success();
}

static LogicalResult emitElementwiseExprToReg(
    Value exprValue, int dstReg, int tmpRegA, int tmpRegB, linalg::GenericOp op,
    linalg::GenericOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
    Location loc, Value tileIdx, std::optional<Value> rowIdx,
    std::optional<unsigned> rowBcastInput) {
  Value dstRegVal = i32Const(rewriter, loc, dstReg);
  Value tmpRegAVal = i32Const(rewriter, loc, tmpRegA);

  if (auto inputIdx = getBodyInputIndex(op, exprValue)) {
    if (rowBcastInput && *inputIdx == *rowBcastInput) {
      if (!rowIdx)
        return failure();
      rewriter.create<UnaryBcastTileOp>(loc, adaptor.getInputs()[*inputIdx],
                                        *rowIdx, dstRegVal, BcastType::Col);
    } else {
      rewriter.create<CopyTileInitOp>(loc, adaptor.getInputs()[*inputIdx]);
      rewriter.create<CopyTileOp>(loc, adaptor.getInputs()[*inputIdx], tileIdx,
                                  dstRegVal);
    }
    return success();
  }

  Operation *defOp = exprValue.getDefiningOp();
  if (!defOp || defOp->getBlock() != &op.getRegion().front())
    return failure();

  if (auto mulOp = dyn_cast<arith::MulFOp>(defOp)) {
    auto lhsConst = getConstFloatValue(mulOp.getLhs());
    auto rhsConst = getConstFloatValue(mulOp.getRhs());
    if (lhsConst && !rhsConst) {
      if (failed(emitElementwiseExprToReg(
              mulOp.getRhs(), dstReg, tmpRegA, tmpRegB, op, adaptor, rewriter,
              loc, tileIdx, rowIdx, rowBcastInput)))
        return failure();
      rewriter.create<MulUnaryTileOp>(loc, dstRegVal,
                                      getScalarBitsFromFloat(rewriter, loc, *lhsConst));
      return success();
    }
    if (rhsConst && !lhsConst) {
      if (failed(emitElementwiseExprToReg(
              mulOp.getLhs(), dstReg, tmpRegA, tmpRegB, op, adaptor, rewriter,
              loc, tileIdx, rowIdx, rowBcastInput)))
        return failure();
      rewriter.create<MulUnaryTileOp>(loc, dstRegVal,
                                      getScalarBitsFromFloat(rewriter, loc, *rhsConst));
      return success();
    }

    if (failed(emitElementwiseExprToReg(mulOp.getLhs(), dstReg, tmpRegA, tmpRegB,
                                        op, adaptor, rewriter, loc, tileIdx, rowIdx,
                                        rowBcastInput)))
      return failure();
    if (failed(emitElementwiseExprToReg(mulOp.getRhs(), tmpRegA, tmpRegB, dstReg,
                                        op, adaptor, rewriter, loc, tileIdx, rowIdx,
                                        rowBcastInput)))
      return failure();
    rewriter.create<MulBinaryTilesOp>(loc, dstRegVal, tmpRegAVal, dstRegVal);
    return success();
  }

  if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
    if (failed(emitElementwiseExprToReg(addOp.getLhs(), dstReg, tmpRegA, tmpRegB,
                                        op, adaptor, rewriter, loc, tileIdx, rowIdx,
                                        rowBcastInput)))
      return failure();
    if (failed(emitElementwiseExprToReg(addOp.getRhs(), tmpRegA, tmpRegB, dstReg,
                                        op, adaptor, rewriter, loc, tileIdx, rowIdx,
                                        rowBcastInput)))
      return failure();
    rewriter.create<AddBinaryTilesOp>(loc, dstRegVal, tmpRegAVal, dstRegVal);
    return success();
  }

  if (auto subOp = dyn_cast<arith::SubFOp>(defOp)) {
    if (failed(emitElementwiseExprToReg(subOp.getLhs(), dstReg, tmpRegA, tmpRegB,
                                        op, adaptor, rewriter, loc, tileIdx, rowIdx,
                                        rowBcastInput)))
      return failure();
    if (failed(emitElementwiseExprToReg(subOp.getRhs(), tmpRegA, tmpRegB, dstReg,
                                        op, adaptor, rewriter, loc, tileIdx, rowIdx,
                                        rowBcastInput)))
      return failure();
    rewriter.create<SubBinaryTilesOp>(loc, dstRegVal, tmpRegAVal, dstRegVal);
    return success();
  }

  if (auto divOp = dyn_cast<arith::DivFOp>(defOp)) {
    if (failed(emitElementwiseExprToReg(divOp.getLhs(), dstReg, tmpRegA, tmpRegB,
                                        op, adaptor, rewriter, loc, tileIdx, rowIdx,
                                        rowBcastInput)))
      return failure();
    if (failed(emitElementwiseExprToReg(divOp.getRhs(), tmpRegA, tmpRegB, dstReg,
                                        op, adaptor, rewriter, loc, tileIdx, rowIdx,
                                        rowBcastInput)))
      return failure();
    rewriter.create<RecipTileOp>(loc, tmpRegAVal);
    rewriter.create<MulBinaryTilesOp>(loc, dstRegVal, tmpRegAVal, dstRegVal);
    return success();
  }

  if (auto powOp = dyn_cast<math::PowFOp>(defOp)) {
    if (!getConstFloatValue(powOp.getLhs()).has_value())
      return failure();
    if (failed(emitElementwiseExprToReg(powOp.getRhs(), dstReg, tmpRegA, tmpRegB,
                                        op, adaptor, rewriter, loc, tileIdx, rowIdx,
                                        rowBcastInput)))
      return failure();
    rewriter.create<ExpTileOp>(loc, dstRegVal);
    return success();
  }

  if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
    Value lhs;
    Value rhs;
    if (!matchMaxSelect(selectOp, lhs, rhs))
      return failure();
    if (failed(emitElementwiseExprToReg(lhs, dstReg, tmpRegA, tmpRegB, op, adaptor,
                                        rewriter, loc, tileIdx, rowIdx,
                                        rowBcastInput)))
      return failure();
    if (failed(emitElementwiseExprToReg(rhs, tmpRegA, tmpRegB, dstReg, op, adaptor,
                                        rewriter, loc, tileIdx, rowIdx,
                                        rowBcastInput)))
      return failure();
    rewriter.create<BinaryMaxTileOp>(loc, dstRegVal, tmpRegAVal, dstRegVal);
    return success();
  }

  return failure();
}

static LogicalResult rewriteReduceGeneric(linalg::GenericOp op,
                                          linalg::GenericOp::Adaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          llvm::DenseMap<Value, int64_t> &waitState) {
  if ((adaptor.getInputs().size() != 1 && adaptor.getInputs().size() != 2) ||
      adaptor.getOutputs().size() != 1)
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
  Value scaleCb =
      adaptor.getInputs().size() == 2 ? adaptor.getInputs()[1] : inCb;
  Value outCb = adaptor.getOutputs()[0];
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

  SmallVector<std::tuple<Value, Value, int64_t>, 2> inputPlans;
  inputPlans.emplace_back(inCb, op.getDpsInputs()[0], numTiles);
  if (adaptor.getInputs().size() == 2)
    inputPlans.emplace_back(scaleCb, op.getDpsInputs()[1], 1);

  for (const auto &plan : inputPlans) {
    Value inputCb = std::get<0>(plan);
    int64_t waitTiles = std::get<2>(plan);
    emitWaitFrontIfNeeded(rewriter, loc, inputCb, waitTiles, waitState);
  }

  CBReserveBackOp::create(rewriter, loc, outCb, outTilesV);

  for (int64_t i = 0; i < rows; ++i) {
    TileRegsAcquireOp::create(rewriter, loc);
    rewriter.create<ReduceInitOp>(loc, inCb, scaleCb, outCb, reduceType,
                                  ReduceDim::Row);
    for (int64_t j = 0; j < cols; ++j) {
      int64_t inTile = i * cols + j;
      rewriter.create<ReduceTileOp>(loc, inCb, scaleCb,
                                    i32Const(rewriter, loc, inTile), zeroI32,
                                    zeroI32, reduceType, ReduceDim::Row);
    }
    rewriter.create<ReduceUninitOp>(loc);
    PackTileOp::create(rewriter, loc, zeroI32, outCb, i32Const(rewriter, loc, i));
    TileRegsReleaseOp::create(rewriter, loc);
  }

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

  rewriter.create<InitSFPUOp>(loc, outCb, outCb);
  if (analysis.needsBinopWithScalar)
    rewriter.create<BinopWithScalarTileInitOp>(loc);
  if (analysis.needsUnaryBcast) {
    unsigned rowInput = *analysis.rowBcastInput;
    unsigned refInput = rowInput;
    for (unsigned i = 0; i < adaptor.getInputs().size(); ++i) {
      if (i != rowInput && analysis.usedInputs.test(i)) {
        refInput = i;
        break;
      }
    }
    rewriter.create<UnaryBcastInitOp>(loc, adaptor.getInputs()[rowInput],
                                      adaptor.getInputs()[refInput],
                                      BcastType::Col);
  }
  if (analysis.needsSubBinary)
    rewriter.create<SubBinaryTilesInitOp>(loc);
  if (analysis.needsAddBinary)
    rewriter.create<AddBinaryTilesInitOp>(loc);
  if (analysis.needsMulBinary)
    rewriter.create<MulBinaryTilesInitOp>(loc);
  if (analysis.needsBinaryMax)
    rewriter.create<BinaryMaxTileInitOp>(loc);
  if (analysis.needsExp)
    rewriter.create<ExpTileInitOp>(loc);
  if (analysis.needsRecip)
    rewriter.create<RecipTileInitOp>(loc);

  LogicalResult result = emitElementwiseTiles(
      rewriter, loc, outCb, analysis.outTiles, [&](int64_t tile) -> LogicalResult {
        Value tileIdx = i32Const(rewriter, loc, tile);
        std::optional<Value> rowIdx;
        if (analysis.rowBcastInput) {
          if (!analysis.colTiles || *analysis.colTiles <= 0)
            return failure();
          rowIdx = i32Const(rewriter, loc, tile / *analysis.colTiles);
        }
        return emitElementwiseExprToReg(
            analysis.yieldValue, /*dstReg=*/0, /*tmpRegA=*/1, /*tmpRegB=*/2, op,
            adaptor, rewriter, loc, tileIdx, rowIdx, analysis.rowBcastInput);
      });
  if (failed(result))
    return failure();

  if (outAliasesInput) {
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
    llvm::errs() << "in0Cb: " << in0Cb << "\n";
    llvm::errs() << "in1Cb: " << in1Cb << "\n";
    llvm::errs() << "outCb: " << outCb << "\n";
    llvm::errs() << "outBuffer: " << outBuffer << "\n";

    // Ensure operands are TTKernel CBs.
    if (!isa<CBType>(in0Cb.getType()) || !isa<CBType>(in1Cb.getType()) ||
        !isa<CBType>(outCb.getType()))
      return failure();

    auto in0Tiles = getNumTilesFromShapedType(op.getInputs()[0].getType());
    auto in1Tiles = getNumTilesFromShapedType(op.getInputs()[1].getType());
    if (!in0Tiles || !in1Tiles)
      return failure();

    // Use the original linalg.matmul input type to query the tensor/memref shape.
    auto lhsShapedType = dyn_cast<ShapedType>(op.getInputs()[0].getType());
    if (!lhsShapedType || !lhsShapedType.hasStaticShape() ||
        lhsShapedType.getRank() != 2)
      return failure();
    ArrayRef<int64_t> lhsShape = lhsShapedType.getShape();
    int32_t rtVal = static_cast<int32_t>(lhsShape[0] / 32);
    int32_t ktVal = static_cast<int32_t>(lhsShape[1] / 32);

    auto rhsShapedType = dyn_cast<ShapedType>(op.getInputs()[1].getType());
    if (!rhsShapedType || !rhsShapedType.hasStaticShape() ||
        rhsShapedType.getRank() != 2)
      return failure();
    ArrayRef<int64_t> rhsShape = rhsShapedType.getShape();
    int32_t ctVal = static_cast<int32_t>(rhsShape[1] / 32);

    auto outShapedType = dyn_cast<ShapedType>(op.getOutputs()[0].getType());
    if (!outShapedType || !outShapedType.hasStaticShape() ||
        outShapedType.getRank() != 2)
      return failure();
    ArrayRef<int64_t> outShape = outShapedType.getShape();
    int32_t ntVal = static_cast<int32_t>(outShape[1] / 32);

    Value zeroI32;
    Value in0TileIdx;
    Value in1TileIdx;
    Value dstTileIdx;
    Value transpose;
    Value ctDim;
    Value rtDim;
    Value ntDim;
    Value ktDim;

    Value in0TileCount = i32Const(rewriter, loc, *in0Tiles);
    Value in1TileCount = i32Const(rewriter, loc, *in1Tiles);
    if (in0Cb == in1Cb) {
      int64_t sharedTiles = std::max(*in0Tiles, *in1Tiles);
      CBWaitFrontOp::create(rewriter, loc, in0Cb,
                            i32Const(rewriter, loc, sharedTiles));
    } else {
      CBWaitFrontOp::create(rewriter, loc, in0Cb, in0TileCount);
      CBWaitFrontOp::create(rewriter, loc, in1Cb, in1TileCount);
    }

    {
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
          loc, /*value=*/0, /*width=*/32);
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
    }

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

      auto outCbType = cast<CBType>(outCb.getType());
      int32_t numTiles = static_cast<int32_t>(outCbType.getNumElements()) / 1024;
      Value outCbNumTiles =
          rewriter.create<arith::ConstantIntOp>(loc, numTiles, 32);
      Value lowerBound = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
      Value step = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);

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

    if (auto nextUse =
            findNextNonViewUseInSameBlock(op.getDpsInits()[0], op.getOperation())) {
      if (isUsedAsLinalgOutput(nextUse->user, nextUse->usedValue)) {
        rewriter.eraseOp(op);
        return success();
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

    CBReserveBackOp::create(rewriter, loc, outCb, i32Const(rewriter, loc, *numTiles));
    rewriter.create<InitSFPUOp>(loc, outCb, outCb);
    rewriter.create<FillTileInitOp>(loc);
    for (int64_t i = 0; i < *numTiles; ++i) {
      TileRegsAcquireOp::create(rewriter, loc);
      rewriter.create<FillTileOp>(loc, i32Const(rewriter, loc, 0), fillValue);
      PackTileOp::create(rewriter, loc, i32Const(rewriter, loc, 0), outCb,
                         i32Const(rewriter, loc, i));
      TileRegsReleaseOp::create(rewriter, loc);
    }
    CBPushBackOp::create(rewriter, loc, outCb, i32Const(rewriter, loc, *numTiles));
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

    CBWaitFrontOp::create(rewriter, loc, inCb, tileCount);
    CBReserveBackOp::create(rewriter, loc, outCb, tileCount);
    rewriter.create<CopyTileInitOp>(loc, inCb);

    for (int64_t i = 0; i < *inTiles; ++i) {
      Value tileIdx = i32Const(rewriter, loc, i);
      TileRegsAcquireOp::create(rewriter, loc);
      rewriter.create<CopyTileOp>(loc, inCb, tileIdx, zero);
      PackTileOp::create(rewriter, loc, zero, outCb, tileIdx);
      TileRegsReleaseOp::create(rewriter, loc);
    }

    CBPushBackOp::create(rewriter, loc, outCb, tileCount);
    rewriter.eraseOp(op);
    return success();
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
    MLIRContext *context) {
  patterns.add<ConvertLinalgFillOp>(typeConverter, context);
  patterns.add<ConvertLinalgCopyOp>(typeConverter, context);
  patterns.add<ConvertLinalgMatmulOp>(typeConverter, context);
  patterns.add<ConvertMemrefCollapseShapeOp>(typeConverter, context);
  patterns.add<ConvertFlashAttentionGenericOp>(typeConverter, context);
}
