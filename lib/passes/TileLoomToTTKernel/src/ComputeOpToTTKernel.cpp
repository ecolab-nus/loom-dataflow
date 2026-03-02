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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "llvm/ADT/STLExtras.h"

// Loom dialect headers for ::loom::CopyOp.
#include "mlir/Interfaces/ViewLikeInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#include <cstdint>
#include <optional>

using namespace mlir;
using namespace tt::ttkernel;

namespace {

static Value stripMemrefCasts(Value value) {
  Value current = value;
  while (auto cast = current.getDefiningOp<memref::CastOp>())
    current = cast.getSource();
  return current;
}

enum class FlashAttentionGenericKind {
  ReduceMaxRow,
  MaxWithScale,
  SubExpBcastCols,
  ReduceSumRow,
  AlphaExp,
  MulAddInplace,
  AccUpdateBcastCols,
  NormalizeDivBcastCols,
};

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

  if (op.getNumLoops() == 3 && op.getNumDpsInputs() == 1) {
    if (bodyHasOp<arith::MaximumFOp>(op))
      return FlashAttentionGenericKind::ReduceMaxRow;
    if (bodyHasOp<arith::AddFOp>(op))
      return FlashAttentionGenericKind::ReduceSumRow;
  }

  if (op.getNumLoops() == 2 && op.getNumDpsInputs() == 2) {
    if (bodyHasOp<arith::CmpFOp>(op) && bodyHasOp<arith::SelectOp>(op))
      return FlashAttentionGenericKind::MaxWithScale;
    if (bodyHasOp<arith::SubFOp>(op) && bodyHasOp<math::PowFOp>(op))
      return FlashAttentionGenericKind::AlphaExp;
  }

  if (op.getNumLoops() == 3 && op.getNumDpsInputs() == 2) {
    if (bodyHasOp<arith::DivFOp>(op))
      return FlashAttentionGenericKind::NormalizeDivBcastCols;
    if (bodyHasOp<arith::SubFOp>(op) && bodyHasOp<math::PowFOp>(op))
      return FlashAttentionGenericKind::SubExpBcastCols;
  }

  if (op.getNumLoops() == 2 && op.getNumDpsInputs() == 3 &&
      bodyHasOp<arith::MulFOp>(op) && bodyHasOp<arith::AddFOp>(op)) {
    return FlashAttentionGenericKind::MulAddInplace;
  }

  if (op.getNumLoops() == 3 && op.getNumDpsInputs() == 3 &&
      bodyHasOp<arith::MulFOp>(op) && bodyHasOp<arith::AddFOp>(op)) {
    return FlashAttentionGenericKind::AccUpdateBcastCols;
  }

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

static Value getOrCreateScalarBitsParam(linalg::GenericOp op, Location loc,
                                        ConversionPatternRewriter &rewriter) {
  for (Operation &inner : op.getRegion().front().without_terminator()) {
    auto trunc = dyn_cast<arith::TruncFOp>(inner);
    if (!trunc)
      continue;
    auto cst = trunc.getIn().getDefiningOp<arith::ConstantFloatOp>();
    if (!cst)
      continue;
    auto floatAttr = dyn_cast<FloatAttr>(cst.getValue());
    if (!floatAttr)
      continue;
    auto f32Value = static_cast<float>(floatAttr.getValueAsDouble());
    Value f32Const = rewriter.create<arith::ConstantFloatOp>(
        loc, rewriter.getF32Type(), llvm::APFloat(f32Value));
    return rewriter.create<arith::BitcastOp>(loc, rewriter.getI32Type(),
                                             f32Const);
  }

  Value oneF32 = rewriter.create<arith::ConstantFloatOp>(
      loc, rewriter.getF32Type(), llvm::APFloat(1.0f));
  return rewriter.create<arith::BitcastOp>(loc, rewriter.getI32Type(), oneF32);
}

static LogicalResult checkBF16Operands(Operation *op, ValueRange operands) {
  for (Value operand : operands) {
    auto shaped = dyn_cast<ShapedType>(operand.getType());
    if (!shaped)
      continue;
    if (!shaped.getElementType().isBF16()) {
      return op->emitOpError(
          "FlashAttention generic lowering currently supports bf16 only");
    }
  }
  return success();
}

static bool hasOutputAlias(Value outCb, ValueRange inCbs) {
  return llvm::is_contained(inCbs, outCb);
}

template <typename BuilderFn>
static LogicalResult emitElementwiseTiles(
    ConversionPatternRewriter &rewriter, Location loc, Value outCb,
    int64_t outTiles, ValueRange inputCbs, BuilderFn &&builder) {
  for (Value inputCb : inputCbs)
    CBWaitFrontOp::create(rewriter, loc, inputCb, i32Const(rewriter, loc, outTiles));

  bool outAliasesInput = hasOutputAlias(outCb, inputCbs);
  if (!outAliasesInput)
    CBReserveBackOp::create(rewriter, loc, outCb, i32Const(rewriter, loc, outTiles));

  for (int64_t i = 0; i < outTiles; ++i) {
    TileRegsAcquireOp::create(rewriter, loc);
    if (failed(builder(i)))
      return failure();
    PackTileOp::create(rewriter, loc, i32Const(rewriter, loc, 0), outCb,
                       i32Const(rewriter, loc, i));
    TileRegsReleaseOp::create(rewriter, loc);
  }

  if (!outAliasesInput)
    CBPushBackOp::create(rewriter, loc, outCb, i32Const(rewriter, loc, outTiles));
  return success();
}

static LogicalResult rewriteReduceGeneric(linalg::GenericOp op,
                                          linalg::GenericOp::Adaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          ReduceType reduceType) {
  if (adaptor.getInputs().size() != 1 || adaptor.getOutputs().size() != 1)
    return failure();

  Value inCb = adaptor.getInputs()[0];
  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(inCb.getType()) || !isa<CBType>(outCb.getType()))
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

  Location loc = op.getLoc();
  Value totalInTiles = i32Const(rewriter, loc, (*bTiles) * (*mTiles) * (*nTiles));
  Value outTilesV = i32Const(rewriter, loc, *outTiles);
  Value zeroI32 = i32Const(rewriter, loc, 0);

  CBWaitFrontOp::create(rewriter, loc, inCb, totalInTiles);
  CBReserveBackOp::create(rewriter, loc, outCb, outTilesV);

  for (int64_t b = 0; b < *bTiles; ++b) {
    for (int64_t m = 0; m < *mTiles; ++m) {
      int64_t rowTile = b * (*mTiles) + m;
      TileRegsAcquireOp::create(rewriter, loc);
      rewriter.create<ReduceInitOp>(loc, inCb, inCb, outCb, reduceType,
                                    ReduceDim::Row);
      for (int64_t n = 0; n < *nTiles; ++n) {
        int64_t inTile = rowTile * (*nTiles) + n;
        rewriter.create<ReduceTileOp>(loc, inCb, inCb, i32Const(rewriter, loc, inTile),
                                      zeroI32, zeroI32, reduceType,
                                      ReduceDim::Row);
      }
      rewriter.create<ReduceUninitOp>(loc);
      PackTileOp::create(rewriter, loc, zeroI32, outCb,
                         i32Const(rewriter, loc, rowTile));
      TileRegsReleaseOp::create(rewriter, loc);
    }
  }

  CBPushBackOp::create(rewriter, loc, outCb, outTilesV);
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult rewriteMaxWithScaleGeneric(
    linalg::GenericOp op, linalg::GenericOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  if (adaptor.getInputs().size() != 2 || adaptor.getOutputs().size() != 1)
    return failure();

  Value lhsCb = adaptor.getInputs()[0];
  Value rhsCb = adaptor.getInputs()[1];
  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(lhsCb.getType()) || !isa<CBType>(rhsCb.getType()) ||
      !isa<CBType>(outCb.getType()))
    return failure();

  auto outTiles = getNumTilesFromShapedType(op.getDpsInits()[0].getType());
  if (!outTiles)
    return failure();

  Location loc = op.getLoc();
  Value scalarBits = getOrCreateScalarBitsParam(op, loc, rewriter);
  rewriter.create<InitSFPUOp>(loc, rhsCb, outCb);
  rewriter.create<BinopWithScalarTileInitOp>(loc);
  rewriter.create<BinaryMaxTileInitOp>(loc);

  auto builder = [&](int64_t tileIdx) -> LogicalResult {
    Value tileI32 = i32Const(rewriter, loc, tileIdx);
    Value dst0 = i32Const(rewriter, loc, 0);
    Value dst1 = i32Const(rewriter, loc, 1);

    rewriter.create<CopyTileInitOp>(loc, rhsCb);
    rewriter.create<CopyTileOp>(loc, rhsCb, tileI32, dst0);
    rewriter.create<MulUnaryTileOp>(loc, dst0, scalarBits);

    rewriter.create<CopyTileInitOp>(loc, lhsCb);
    rewriter.create<CopyTileOp>(loc, lhsCb, tileI32, dst1);
    rewriter.create<BinaryMaxTileOp>(loc, dst1, dst0, dst0);
    return success();
  };

  if (failed(emitElementwiseTiles(rewriter, loc, outCb, *outTiles,
                                  ValueRange{lhsCb, rhsCb}, builder)))
    return failure();
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult rewriteSubExpBcastColsGeneric(
    linalg::GenericOp op, linalg::GenericOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  if (adaptor.getInputs().size() != 2 || adaptor.getOutputs().size() != 1)
    return failure();

  Value inCb = adaptor.getInputs()[0];
  Value bcastCb = adaptor.getInputs()[1];
  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(inCb.getType()) || !isa<CBType>(bcastCb.getType()) ||
      !isa<CBType>(outCb.getType()))
    return failure();

  auto inType = dyn_cast<ShapedType>(op.getDpsInputs()[0].getType());
  if (!inType || inType.getRank() != 3)
    return failure();
  auto bTiles = getTileDim(inType, 0);
  auto mTiles = getTileDim(inType, 1);
  auto nTiles = getTileDim(inType, 2);
  auto outTiles = getNumTilesFromShapedType(op.getDpsInits()[0].getType());
  auto bcastTiles = getNumTilesFromShapedType(op.getDpsInputs()[1].getType());
  if (!bTiles || !mTiles || !nTiles || !outTiles || !bcastTiles)
    return failure();

  Location loc = op.getLoc();
  Value scalarBits = getOrCreateScalarBitsParam(op, loc, rewriter);
  rewriter.create<InitSFPUOp>(loc, inCb, outCb);
  rewriter.create<BinopWithScalarTileInitOp>(loc);
  rewriter.create<UnaryBcastInitOp>(loc, bcastCb, outCb, BcastType::Col);
  rewriter.create<SubBinaryTilesInitOp>(loc);
  rewriter.create<ExpTileInitOp>(loc);

  CBWaitFrontOp::create(rewriter, loc, inCb, i32Const(rewriter, loc, *outTiles));
  CBWaitFrontOp::create(rewriter, loc, bcastCb, i32Const(rewriter, loc, *bcastTiles));
  if (!hasOutputAlias(outCb, ValueRange{inCb, bcastCb}))
    CBReserveBackOp::create(rewriter, loc, outCb, i32Const(rewriter, loc, *outTiles));

  for (int64_t b = 0; b < *bTiles; ++b) {
    for (int64_t m = 0; m < *mTiles; ++m) {
      int64_t rowTile = b * (*mTiles) + m;
      for (int64_t n = 0; n < *nTiles; ++n) {
        int64_t outTile = rowTile * (*nTiles) + n;
        TileRegsAcquireOp::create(rewriter, loc);
        rewriter.create<CopyTileInitOp>(loc, inCb);
        rewriter.create<CopyTileOp>(loc, inCb, i32Const(rewriter, loc, outTile),
                                    i32Const(rewriter, loc, 0));
        rewriter.create<MulUnaryTileOp>(loc, i32Const(rewriter, loc, 0), scalarBits);
        rewriter.create<UnaryBcastTileOp>(loc, bcastCb, i32Const(rewriter, loc, rowTile),
                                          i32Const(rewriter, loc, 1),
                                          BcastType::Col);
        rewriter.create<SubBinaryTilesOp>(loc, i32Const(rewriter, loc, 0),
                                          i32Const(rewriter, loc, 1),
                                          i32Const(rewriter, loc, 0));
        rewriter.create<ExpTileOp>(loc, i32Const(rewriter, loc, 0));
        PackTileOp::create(rewriter, loc, i32Const(rewriter, loc, 0), outCb,
                           i32Const(rewriter, loc, outTile));
        TileRegsReleaseOp::create(rewriter, loc);
      }
    }
  }

  if (!hasOutputAlias(outCb, ValueRange{inCb, bcastCb}))
    CBPushBackOp::create(rewriter, loc, outCb, i32Const(rewriter, loc, *outTiles));
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult rewriteAlphaExpGeneric(linalg::GenericOp op,
                                            linalg::GenericOp::Adaptor adaptor,
                                            ConversionPatternRewriter &rewriter) {
  if (adaptor.getInputs().size() != 2 || adaptor.getOutputs().size() != 1)
    return failure();

  Value lhsCb = adaptor.getInputs()[0];
  Value rhsCb = adaptor.getInputs()[1];
  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(lhsCb.getType()) || !isa<CBType>(rhsCb.getType()) ||
      !isa<CBType>(outCb.getType()))
    return failure();

  auto outTiles = getNumTilesFromShapedType(op.getDpsInits()[0].getType());
  if (!outTiles)
    return failure();

  Location loc = op.getLoc();
  rewriter.create<InitSFPUOp>(loc, lhsCb, outCb);
  rewriter.create<SubBinaryTilesInitOp>(loc);
  rewriter.create<ExpTileInitOp>(loc);

  auto builder = [&](int64_t tileIdx) -> LogicalResult {
    Value tileI32 = i32Const(rewriter, loc, tileIdx);
    Value dst0 = i32Const(rewriter, loc, 0);
    Value dst1 = i32Const(rewriter, loc, 1);
    rewriter.create<CopyTileInitOp>(loc, lhsCb);
    rewriter.create<CopyTileOp>(loc, lhsCb, tileI32, dst0);
    rewriter.create<CopyTileInitOp>(loc, rhsCb);
    rewriter.create<CopyTileOp>(loc, rhsCb, tileI32, dst1);
    rewriter.create<SubBinaryTilesOp>(loc, dst0, dst1, dst0);
    rewriter.create<ExpTileOp>(loc, dst0);
    return success();
  };

  if (failed(emitElementwiseTiles(rewriter, loc, outCb, *outTiles,
                                  ValueRange{lhsCb, rhsCb}, builder)))
    return failure();
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult rewriteMulAddInplaceGeneric(
    linalg::GenericOp op, linalg::GenericOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  if (adaptor.getInputs().size() != 3 || adaptor.getOutputs().size() != 1)
    return failure();

  Value in0Cb = adaptor.getInputs()[0];
  Value in1Cb = adaptor.getInputs()[1];
  Value in2Cb = adaptor.getInputs()[2];
  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(in0Cb.getType()) || !isa<CBType>(in1Cb.getType()) ||
      !isa<CBType>(in2Cb.getType()) || !isa<CBType>(outCb.getType()))
    return failure();

  auto outTiles = getNumTilesFromShapedType(op.getDpsInits()[0].getType());
  if (!outTiles)
    return failure();

  Location loc = op.getLoc();
  rewriter.create<InitSFPUOp>(loc, in0Cb, outCb);
  rewriter.create<MulBinaryTilesInitOp>(loc);
  rewriter.create<AddBinaryTilesInitOp>(loc);

  auto builder = [&](int64_t tileIdx) -> LogicalResult {
    Value tileI32 = i32Const(rewriter, loc, tileIdx);
    Value dst0 = i32Const(rewriter, loc, 0);
    Value dst1 = i32Const(rewriter, loc, 1);
    Value dst2 = i32Const(rewriter, loc, 2);
    rewriter.create<CopyTileInitOp>(loc, in0Cb);
    rewriter.create<CopyTileOp>(loc, in0Cb, tileI32, dst0);
    rewriter.create<CopyTileInitOp>(loc, in1Cb);
    rewriter.create<CopyTileOp>(loc, in1Cb, tileI32, dst1);
    rewriter.create<MulBinaryTilesOp>(loc, dst0, dst1, dst0);
    rewriter.create<CopyTileInitOp>(loc, in2Cb);
    rewriter.create<CopyTileOp>(loc, in2Cb, tileI32, dst2);
    rewriter.create<AddBinaryTilesOp>(loc, dst0, dst2, dst0);
    return success();
  };

  if (failed(emitElementwiseTiles(rewriter, loc, outCb, *outTiles,
                                  ValueRange{in0Cb, in1Cb, in2Cb}, builder)))
    return failure();
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult rewriteAccUpdateBcastColsGeneric(
    linalg::GenericOp op, linalg::GenericOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  if (adaptor.getInputs().size() != 3 || adaptor.getOutputs().size() != 1)
    return failure();

  Value pvCb = adaptor.getInputs()[0];
  Value accCb = adaptor.getInputs()[1];
  Value alphaCb = adaptor.getInputs()[2];
  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(pvCb.getType()) || !isa<CBType>(accCb.getType()) ||
      !isa<CBType>(alphaCb.getType()) || !isa<CBType>(outCb.getType()))
    return failure();

  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  if (!outType || outType.getRank() != 3)
    return failure();
  auto bTiles = getTileDim(outType, 0);
  auto mTiles = getTileDim(outType, 1);
  auto nTiles = getTileDim(outType, 2);
  auto outTiles = getNumTilesFromShapedType(outType);
  auto alphaTiles = getNumTilesFromShapedType(op.getDpsInputs()[2].getType());
  if (!bTiles || !mTiles || !nTiles || !outTiles || !alphaTiles)
    return failure();

  Location loc = op.getLoc();
  rewriter.create<InitSFPUOp>(loc, accCb, outCb);
  rewriter.create<UnaryBcastInitOp>(loc, alphaCb, outCb, BcastType::Col);
  rewriter.create<MulBinaryTilesInitOp>(loc);
  rewriter.create<AddBinaryTilesInitOp>(loc);

  CBWaitFrontOp::create(rewriter, loc, pvCb, i32Const(rewriter, loc, *outTiles));
  CBWaitFrontOp::create(rewriter, loc, accCb, i32Const(rewriter, loc, *outTiles));
  CBWaitFrontOp::create(rewriter, loc, alphaCb, i32Const(rewriter, loc, *alphaTiles));
  if (!hasOutputAlias(outCb, ValueRange{pvCb, accCb, alphaCb}))
    CBReserveBackOp::create(rewriter, loc, outCb, i32Const(rewriter, loc, *outTiles));

  for (int64_t b = 0; b < *bTiles; ++b) {
    for (int64_t m = 0; m < *mTiles; ++m) {
      int64_t rowTile = b * (*mTiles) + m;
      for (int64_t n = 0; n < *nTiles; ++n) {
        int64_t outTile = rowTile * (*nTiles) + n;
        TileRegsAcquireOp::create(rewriter, loc);
        rewriter.create<CopyTileInitOp>(loc, accCb);
        rewriter.create<CopyTileOp>(loc, accCb, i32Const(rewriter, loc, outTile),
                                    i32Const(rewriter, loc, 0));
        rewriter.create<UnaryBcastTileOp>(loc, alphaCb, i32Const(rewriter, loc, rowTile),
                                          i32Const(rewriter, loc, 1),
                                          BcastType::Col);
        rewriter.create<MulBinaryTilesOp>(loc, i32Const(rewriter, loc, 0),
                                          i32Const(rewriter, loc, 1),
                                          i32Const(rewriter, loc, 0));
        rewriter.create<CopyTileInitOp>(loc, pvCb);
        rewriter.create<CopyTileOp>(loc, pvCb, i32Const(rewriter, loc, outTile),
                                    i32Const(rewriter, loc, 2));
        rewriter.create<AddBinaryTilesOp>(loc, i32Const(rewriter, loc, 2),
                                          i32Const(rewriter, loc, 0),
                                          i32Const(rewriter, loc, 0));
        PackTileOp::create(rewriter, loc, i32Const(rewriter, loc, 0), outCb,
                           i32Const(rewriter, loc, outTile));
        TileRegsReleaseOp::create(rewriter, loc);
      }
    }
  }

  if (!hasOutputAlias(outCb, ValueRange{pvCb, accCb, alphaCb}))
    CBPushBackOp::create(rewriter, loc, outCb, i32Const(rewriter, loc, *outTiles));
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult rewriteNormalizeDivBcastColsGeneric(
    linalg::GenericOp op, linalg::GenericOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  if (adaptor.getInputs().size() != 2 || adaptor.getOutputs().size() != 1)
    return failure();

  Value accCb = adaptor.getInputs()[0];
  Value liCb = adaptor.getInputs()[1];
  Value outCb = adaptor.getOutputs()[0];
  if (!isa<CBType>(accCb.getType()) || !isa<CBType>(liCb.getType()) ||
      !isa<CBType>(outCb.getType()))
    return failure();

  auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
  if (!outType || outType.getRank() != 3)
    return failure();
  auto bTiles = getTileDim(outType, 0);
  auto mTiles = getTileDim(outType, 1);
  auto nTiles = getTileDim(outType, 2);
  auto outTiles = getNumTilesFromShapedType(outType);
  auto liTiles = getNumTilesFromShapedType(op.getDpsInputs()[1].getType());
  if (!bTiles || !mTiles || !nTiles || !outTiles || !liTiles)
    return failure();

  Location loc = op.getLoc();
  rewriter.create<InitSFPUOp>(loc, liCb, liCb);
  rewriter.create<RecipTileInitOp>(loc);
  CBWaitFrontOp::create(rewriter, loc, liCb, i32Const(rewriter, loc, *liTiles));
  for (int64_t t = 0; t < *liTiles; ++t) {
    TileRegsAcquireOp::create(rewriter, loc);
    rewriter.create<CopyTileInitOp>(loc, liCb);
    rewriter.create<CopyTileOp>(loc, liCb, i32Const(rewriter, loc, t),
                                i32Const(rewriter, loc, 0));
    rewriter.create<RecipTileOp>(loc, i32Const(rewriter, loc, 0));
    PackTileOp::create(rewriter, loc, i32Const(rewriter, loc, 0), liCb,
                       i32Const(rewriter, loc, t));
    TileRegsReleaseOp::create(rewriter, loc);
  }

  rewriter.create<InitSFPUOp>(loc, accCb, outCb);
  rewriter.create<UnaryBcastInitOp>(loc, liCb, outCb, BcastType::Col);
  rewriter.create<MulBinaryTilesInitOp>(loc);
  CBWaitFrontOp::create(rewriter, loc, accCb, i32Const(rewriter, loc, *outTiles));
  CBWaitFrontOp::create(rewriter, loc, liCb, i32Const(rewriter, loc, *liTiles));
  CBReserveBackOp::create(rewriter, loc, outCb, i32Const(rewriter, loc, *outTiles));

  for (int64_t b = 0; b < *bTiles; ++b) {
    for (int64_t m = 0; m < *mTiles; ++m) {
      int64_t rowTile = b * (*mTiles) + m;
      for (int64_t n = 0; n < *nTiles; ++n) {
        int64_t outTile = rowTile * (*nTiles) + n;
        TileRegsAcquireOp::create(rewriter, loc);
        rewriter.create<CopyTileInitOp>(loc, accCb);
        rewriter.create<CopyTileOp>(loc, accCb, i32Const(rewriter, loc, outTile),
                                    i32Const(rewriter, loc, 0));
        rewriter.create<UnaryBcastTileOp>(loc, liCb, i32Const(rewriter, loc, rowTile),
                                          i32Const(rewriter, loc, 1),
                                          BcastType::Col);
        rewriter.create<MulBinaryTilesOp>(loc, i32Const(rewriter, loc, 0),
                                          i32Const(rewriter, loc, 1),
                                          i32Const(rewriter, loc, 0));
        PackTileOp::create(rewriter, loc, i32Const(rewriter, loc, 0), outCb,
                           i32Const(rewriter, loc, outTile));
        TileRegsReleaseOp::create(rewriter, loc);
      }
    }
  }
  CBPushBackOp::create(rewriter, loc, outCb, i32Const(rewriter, loc, *outTiles));
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

    // Ensure operands are TTKernel CBs.
    if (!isa<CBType>(in0Cb.getType()) || !isa<CBType>(in1Cb.getType()) ||
        !isa<CBType>(outCb.getType()))
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

    rewriter.create<ExperimentalMatmulBlockOp>(
        loc, TypeRange{},
        ValueRange{in0Cb, in1Cb, in0TileIdx, in1TileIdx, dstTileIdx, transpose,
                   ctDim, rtDim, ktDim, ntDim});

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

class ConvertLoomCopyOp : public OpConversionPattern<::loom::CopyOp> {
public:
  ConvertLoomCopyOp(TypeConverter &typeConverter, MLIRContext *context,
                    std::shared_ptr<mlir::loom::CompileArgTracker> tracker)
      : OpConversionPattern<::loom::CopyOp>(typeConverter, context),
        tracker(std::move(tracker)) {}

  LogicalResult
  matchAndRewrite(::loom::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value inCb = adaptor.getSource();

    // Prefer tracker lookup for DRAM->L1 copies so compute kernels reuse the
    // same argument-to-CB mapping as memory lowering, even when adaptor source
    // is already a type-materialized CB via unrealized_conversion_cast.
    if (auto sourceRC = op.getSource().getDefiningOp<memref::ReinterpretCastOp>()) {
      Value inputMemref = stripMemrefCasts(sourceRC.getSource());
      if (tracker) {
        if (Value trackedCb = tracker->getCB(inputMemref))
          inCb = trackedCb;
      }
    }

    if (!isa<CBType>(inCb.getType()))
      return failure();

    Location loc = op.getLoc();
    auto srcMemSpace = op.getSrcMemSpace();
    bool isSrcDRAM =
        srcMemSpace && srcMemSpace->getRootReference().getValue() == "DRAM";
    if (isSrcDRAM) {
      auto inCbType = cast<CBType>(inCb.getType());
      Value inNumInputTilesValue = rewriter.create<arith::ConstantIntOp>(
          loc, static_cast<int32_t>(inCbType.getNumElements()) / 1024, 32);
      CBWaitFrontOp::create(rewriter, loc, inCb, inNumInputTilesValue);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  std::shared_ptr<mlir::loom::CompileArgTracker> tracker;
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
    if (!mlir::loom::isSupportedFlashAttentionGeneric(op))
      return failure();

    auto kind = classifyFlashAttentionGeneric(op);
    if (!kind)
      return failure();

    if (failed(checkBF16Operands(op, op->getOperands())))
      return failure();

    switch (*kind) {
    case FlashAttentionGenericKind::ReduceMaxRow:
      return rewriteReduceGeneric(op, adaptor, rewriter, ReduceType::Max);
    case FlashAttentionGenericKind::ReduceSumRow:
      return rewriteReduceGeneric(op, adaptor, rewriter, ReduceType::Sum);
    case FlashAttentionGenericKind::MaxWithScale:
      return rewriteMaxWithScaleGeneric(op, adaptor, rewriter);
    case FlashAttentionGenericKind::SubExpBcastCols:
      return rewriteSubExpBcastColsGeneric(op, adaptor, rewriter);
    case FlashAttentionGenericKind::AlphaExp:
      return rewriteAlphaExpGeneric(op, adaptor, rewriter);
    case FlashAttentionGenericKind::MulAddInplace:
      return rewriteMulAddInplaceGeneric(op, adaptor, rewriter);
    case FlashAttentionGenericKind::AccUpdateBcastCols:
      return rewriteAccUpdateBcastColsGeneric(op, adaptor, rewriter);
    case FlashAttentionGenericKind::NormalizeDivBcastCols:
      return rewriteNormalizeDivBcastColsGeneric(op, adaptor, rewriter);
    }

    return failure();
  }
};

} // namespace

bool mlir::loom::isSupportedFlashAttentionGeneric(linalg::GenericOp op) {
  return isComputeKernel(op.getOperation()) &&
         classifyFlashAttentionGeneric(op).has_value();
}

void mlir::loom::populateComputeOpConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *context, std::shared_ptr<CompileArgTracker> tracker) {
  patterns.add<ConvertLinalgFillOp>(typeConverter, context);
  patterns.add<ConvertLoomCopyOp>(typeConverter, context, tracker);
  patterns.add<ConvertLinalgMatmulOp>(typeConverter, context);
  patterns.add<ConvertMemrefCollapseShapeOp>(typeConverter, context);
  patterns.add<ConvertFlashAttentionGenericOp>(typeConverter, context);
}
