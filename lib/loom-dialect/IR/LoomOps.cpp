//===- LoomOps.cpp - LOOM Dialect Operations -----------------------------===//
//
// Implementation of the LOOM operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "LoomDialect.h.inc"
#include "LoomEnums.h.inc"
#include "llvm/ADT/TypeSwitch.h"
#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.h.inc"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace loom;

#include "LoomDialect.cpp.inc"

#include "LoomEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.cpp.inc"

#include "LoomInterfaces.cpp.inc"

void LoomDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LoomOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "LoomAttributes.cpp.inc"
      >();
}

// Custom assembly format helpers are provided by mlir::parseDynamicIndexList
// and mlir::printDynamicIndexList from ViewLikeInterface.h

//===----------------------------------------------------------------------===//
// Loom Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "LoomOps.cpp.inc"


//===----------------------------------------------------------------------===//
// SubviewOp Type Inference
//===----------------------------------------------------------------------===//

MemRefType loom::SubviewOp::inferResultType(MemRefType sourceType,
                                            ArrayRef<int64_t> staticOffsets,
                                            ArrayRef<int64_t> staticSizes,
                                            ArrayRef<int64_t> staticStrides) {

  // Extract element type and memory space from source
  Type elementType = sourceType.getElementType();
  Attribute memorySpace = sourceType.getMemorySpace();

  // Compute result shape from static sizes
  // Dynamic dimensions use ShapedType::kDynamic
  SmallVector<int64_t> resultShape(staticSizes.begin(), staticSizes.end());

  // Compute strides for the result memref
  // For a view, strides come from the source memref
  // Result stride[i] = source_stride[i] * view_stride[i]
  auto sourceLayout = sourceType.getLayout();
  SmallVector<int64_t> resultStrides;
  int64_t resultOffset = 0;

  if (auto stridedLayout = dyn_cast<StridedLayoutAttr>(sourceLayout)) {
    ArrayRef<int64_t> sourceStrides = stridedLayout.getStrides();
    resultOffset = stridedLayout.getOffset();

    // Compute new offset: offset += sum(static_offset[i] * source_stride[i])
    for (size_t i = 0; i < staticOffsets.size(); ++i) {
      if (staticOffsets[i] != ShapedType::kDynamic &&
          sourceStrides[i] != ShapedType::kDynamic &&
          resultOffset != ShapedType::kDynamic) {
        resultOffset += staticOffsets[i] * sourceStrides[i];
      } else {
        resultOffset = ShapedType::kDynamic;
      }
    }

    // Result strides inherit source strides (view strides are access
    // multipliers)
    for (size_t i = 0; i < staticStrides.size(); ++i) {
      if (staticStrides[i] != ShapedType::kDynamic &&
          sourceStrides[i] != ShapedType::kDynamic) {
        resultStrides.push_back(sourceStrides[i] * staticStrides[i]);
      } else {
        resultStrides.push_back(ShapedType::kDynamic);
      }
    }
  } else {
    // Assume default row-major layout for source
    // Compute strides from source shape
    int64_t stride = 1;
    for (int i = sourceType.getRank() - 1; i >= 0; --i) {
      int64_t currentStride = stride;
      if (i < (int)staticStrides.size()) {
        if (staticStrides[i] != ShapedType::kDynamic &&
            currentStride != ShapedType::kDynamic) {
          resultStrides.insert(resultStrides.begin(),
                               currentStride * staticStrides[i]);
        } else {
          resultStrides.insert(resultStrides.begin(), ShapedType::kDynamic);
        }
      }
      if (sourceType.getDimSize(i) != ShapedType::kDynamic &&
          stride != ShapedType::kDynamic) {
        stride *= sourceType.getDimSize(i);
      } else {
        stride = ShapedType::kDynamic;
      }
    }

    // Keep offset conservative when source layout is implicit identity.
    // This matches verifier expectations for view-like ops in this case.
    resultOffset = ShapedType::kDynamic;
  }

  auto layout = StridedLayoutAttr::get(sourceType.getContext(), resultOffset,
                                       resultStrides);

  return MemRefType::get(resultShape, elementType, layout, memorySpace);
}

MemRefType loom::SubviewOp::inferResultType(MemRefType sourceType,
                                            ArrayRef<int64_t> staticOffsets,
                                            ArrayRef<int64_t> staticSizes,
                                            ArrayRef<int64_t> staticStrides,
                                            ArrayRef<int64_t> targetShape) {
  // Compute the full-rank result type first.
  MemRefType fullType =
      inferResultType(sourceType, staticOffsets, staticSizes, staticStrides);

  if ((int64_t)targetShape.size() == fullType.getRank())
    return fullType;

  // Extract only the strides corresponding to retained positions.
  // A position is dropped when its static size is exactly 1.
  auto stridedLayout = cast<StridedLayoutAttr>(fullType.getLayout());
  ArrayRef<int64_t> fullStrides = stridedLayout.getStrides();
  int64_t fullOffset = stridedLayout.getOffset();

  SmallVector<int64_t> retainedStrides;
  SmallVector<int64_t> retainedShape;
  for (size_t i = 0; i < staticSizes.size(); ++i) {
    if (staticSizes[i] == ShapedType::kDynamic || staticSizes[i] != 1) {
      retainedStrides.push_back(fullStrides[i]);
      retainedShape.push_back(staticSizes[i]);
    }
  }

  auto layout = StridedLayoutAttr::get(sourceType.getContext(), fullOffset,
                                       retainedStrides);
  return MemRefType::get(retainedShape, fullType.getElementType(), layout,
                         fullType.getMemorySpace());
}

::mlir::RankedTensorType
loom::CopyToTensorOp::inferResultType(::mlir::MemRefType sourceType) {
  return RankedTensorType::get(sourceType.getShape(),
                               sourceType.getElementType());
}

::mlir::RankedTensorType
loom::InitTensorOp::inferResultType(::llvm::ArrayRef<int64_t> staticSizes,
                                    ::mlir::Type elementType) {
  return RankedTensorType::get(staticSizes, elementType);
}

::mlir::MemRefType
loom::AllocOp::inferResultType(::llvm::ArrayRef<int64_t> staticSizes,
                               ::mlir::Type elementType) {
  return MemRefType::get(staticSizes, elementType);
}

void loom::ReduceSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void loom::GatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void loom::BroadcastOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void loom::SyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void loom::CopyToTensorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void loom::CopyFromTensorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void loom::CopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}
void loom::SemaphoreTakeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Declare a virtual MemAlloc effect to prevent CSE from merging
  // two semaphore ops that share the same source alloc.
  effects.emplace_back(MemoryEffects::Allocate::get());
}

void loom::SemaphoreGiveOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Free::get());
}

void loom::MatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void loom::BatchMatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

//===----------------------------------------------------------------------===//
// ViewOp Canonicalizers
//===----------------------------------------------------------------------===//

namespace {

bool foldConstantDimensions(SmallVectorImpl<OpFoldResult> &mixedValues) {
  bool changed = false;
  for (OpFoldResult &ofp : mixedValues) {
    if (auto ssaValue = ofp.dyn_cast<Value>()) {
      IntegerAttr attr;
      if (matchPattern(ssaValue, m_Constant(&attr))) {
        ofp = attr;
        changed = true;
      }
    }
  }
  return changed;
}

struct FoldSubviewConstants : public OpRewritePattern<SubviewOp> {
  using OpRewritePattern<SubviewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubviewOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult, 4> offsets = op.getMixedOffsets();
    SmallVector<OpFoldResult, 4> sizes = op.getMixedSizes();
    SmallVector<OpFoldResult, 4> strides = op.getMixedStrides();

    bool changed = false;
    changed |= foldConstantDimensions(offsets);
    changed |= foldConstantDimensions(sizes);
    changed |= foldConstantDimensions(strides);

    if (!changed)
      return failure();

    // Reconstruct the op with new mixed values
    SmallVector<Value, 4> dynamicOffsets, dynamicSizes, dynamicStrides;
    SmallVector<int64_t, 4> staticOffsets, staticSizes, staticStrides;
    dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
    dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
    dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

    // Preserve the original result rank (the op may be rank-reducing).
    auto origResultType = cast<MemRefType>(op.getType());
    auto newResultType = SubviewOp::inferResultType(
        op.getSourceType(), staticOffsets, staticSizes, staticStrides,
        origResultType.getShape());

    rewriter.replaceOpWithNewOp<SubviewOp>(
        op, newResultType, op.getSource(), dynamicOffsets, dynamicSizes,
        dynamicStrides, rewriter.getDenseI64ArrayAttr(staticOffsets),
        rewriter.getDenseI64ArrayAttr(staticSizes),
        rewriter.getDenseI64ArrayAttr(staticStrides),
        op.getSequentialReuseAttr(), op.getSpatialReuseAttr(),
        op.getTemporalReuseAttr());
    return success();
  }
};

struct FoldAllocConstants : public OpRewritePattern<AllocOp> {
  using OpRewritePattern<AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult, 4> sizes = op.getMixedSizes();

    bool constantsFolded = foldConstantDimensions(sizes);

    SmallVector<Value, 4> dynamicSizes;
    SmallVector<int64_t, 4> staticSizes;
    dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);

    // Check if all sizes are now static
    bool allStatic = llvm::all_of(
        staticSizes, [](int64_t s) { return s != ShapedType::kDynamic; });

    auto resultType = llvm::dyn_cast<MemRefType>(op.getType());
    if (!resultType)
      return failure();

    // Infer new result type if all sizes are static but current type is dynamic
    bool needsTypeUpdate = allStatic && !resultType.hasStaticShape();

    if (!constantsFolded && !needsTypeUpdate)
      return failure();

    Type newResultType = resultType;
    if (needsTypeUpdate) {
      newResultType =
          AllocOp::inferResultType(staticSizes, resultType.getElementType());
    }

    auto newOp = rewriter.create<AllocOp>(
        op.getLoc(), newResultType, dynamicSizes,
        rewriter.getDenseI64ArrayAttr(staticSizes), op.getAlignmentAttr(),
        op.getBufferCountAttr(), op.getMemoryAttr());

    // If type changed, insert a cast to keep IR valid for other users.
    if (newResultType != resultType) {
      rewriter.replaceOpWithNewOp<memref::CastOp>(op, resultType,
                                                  newOp.getResult());
    } else {
      rewriter.replaceOp(op, newOp.getResult());
    }
    return success();
  }
};

struct FoldInitTensorConstants : public OpRewritePattern<InitTensorOp> {
  using OpRewritePattern<InitTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InitTensorOp op,
                                PatternRewriter &rewriter) const override {
    Value buffer = op.getBuffer();
    // Look through memref.cast for the buffer
    if (auto castOp = buffer.getDefiningOp<memref::CastOp>()) {
      buffer = castOp.getSource();
    }

    SmallVector<OpFoldResult, 4> sizes = op.getMixedSizes();
    bool constantsFolded = foldConstantDimensions(sizes);

    SmallVector<Value, 4> dynamicSizes;
    SmallVector<int64_t, 4> staticSizes;
    dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);

    // Check if all sizes are now static
    bool allStatic = llvm::all_of(
        staticSizes, [](int64_t s) { return s != ShapedType::kDynamic; });

    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    // Infer new result type if all sizes are static but current type is dynamic
    bool needsTypeUpdate = allStatic && !resultType.hasStaticShape();

    // Also check if we can unwrap the buffer cast even if no sizes changed
    bool needsBufferUpdate = (buffer != op.getBuffer());

    if (!constantsFolded && !needsTypeUpdate && !needsBufferUpdate)
      return failure();

    Type newResultType = resultType;
    if (needsTypeUpdate) {
      newResultType = InitTensorOp::inferResultType(
          staticSizes, resultType.getElementType());
    }

    auto newOp = rewriter.create<InitTensorOp>(
        op.getLoc(), newResultType, buffer, dynamicSizes,
        rewriter.getDenseI64ArrayAttr(staticSizes));

    // If type changed, insert a cast to keep IR valid for other users.
    if (newResultType != resultType) {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                  newOp.getResult());
    } else {
      rewriter.replaceOp(op, newOp.getResult());
    }
    return success();
  }
};

/// Pattern to update scf.for iter_args and results when init args are
/// staticized.
struct StaticizeScfFor : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    bool needsUpdate = false;
    SmallVector<Type> newIterTypes;
    SmallVector<Value> newInits;

    for (auto [initArg, result] :
         llvm::zip(op.getInitArgs(), op.getResults())) {
      Value source = initArg;
      Type sourceType = initArg.getType();

      if (auto cast = initArg.getDefiningOp<tensor::CastOp>()) {
        source = cast.getSource();
        sourceType = source.getType();
      }

      auto rankedSourceType = llvm::dyn_cast<RankedTensorType>(sourceType);
      auto rankedResultType =
          llvm::dyn_cast<RankedTensorType>(result.getType());

      if (rankedSourceType && rankedResultType &&
          rankedSourceType.hasStaticShape() &&
          !rankedResultType.hasStaticShape()) {
        needsUpdate = true;
        newIterTypes.push_back(rankedSourceType);
        newInits.push_back(source);
      } else {
        newIterTypes.push_back(result.getType());
        newInits.push_back(initArg);
      }
    }

    if (!needsUpdate)
      return failure();

    // Create new for loop with updated types. scf.for uses SSA values for
    // lower/upper/step.
    auto newForOp = rewriter.create<scf::ForOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
        newInits);

    // Move the body of the old loop to the new one, overwriting the default
    // entry block (with its auto-inserted scf.yield) that ForOp::create made.
    newForOp.getRegion().takeBody(op.getRegion());

    // Update block argument types for the iteration arguments
    for (auto [idx, blockArg] : llvm::enumerate(newForOp.getRegionIterArgs())) {
      blockArg.setType(newIterTypes[idx]);
    }

    // Update the yield inside the loop to use the static types
    auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    SmallVector<Value> newYieldOperands;
    for (auto [idx, yieldVal] : llvm::enumerate(yieldOp.getOperands())) {
      if (yieldVal.getType() != newIterTypes[idx]) {
        bool castFound = false;
        if (auto castOp = yieldVal.getDefiningOp<tensor::CastOp>()) {
          if (castOp.getSource().getType() == newIterTypes[idx]) {
            newYieldOperands.push_back(castOp.getSource());
            castFound = true;
          }
        }
        if (!castFound) {
          // If no cast found, insert one to the expected static type
          auto castOp = rewriter.create<tensor::CastOp>(
              yieldOp.getLoc(), newIterTypes[idx], yieldVal);
          newYieldOperands.push_back(castOp.getResult());
        }
      } else {
        newYieldOperands.push_back(yieldVal);
      }
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newYieldOperands);

    // Replace old results with casts from new results
    rewriter.setInsertionPointAfter(newForOp);
    SmallVector<Value> replacedResults;
    for (auto [idx, result] : llvm::enumerate(newForOp.getResults())) {
      Type targetType = op.getResults()[idx].getType();
      if (result.getType() != targetType) {
        auto castOp =
            rewriter.create<tensor::CastOp>(op.getLoc(), targetType, result);
        replacedResults.push_back(castOp.getResult());
      } else {
        replacedResults.push_back(result);
      }
    }

    rewriter.replaceOp(op, replacedResults);
    return success();
  }
};

/// Pattern to remove redundant casts before copy_from_tensor.
struct StaticizeCopyFromTensor : public OpRewritePattern<CopyFromTensorOp> {
  using OpRewritePattern<CopyFromTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyFromTensorOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;

    // 1. Unwrap tensor cast for source
    auto source = op.getSourceTensor();
    if (auto cast = source.getDefiningOp<tensor::CastOp>()) {
      auto castSource = cast.getSource();
      auto castSourceType =
          llvm::dyn_cast<RankedTensorType>(castSource.getType());
      if (castSourceType && castSourceType.hasStaticShape()) {
        rewriter.modifyOpInPlace(
            op, [&]() { op.getSourceTensorMutable().assign(castSource); });
        changed = true;
      }
    }

    // 2. Unwrap memref cast for target view
    auto targetView = op.getTargetView();
    if (auto cast = targetView.getDefiningOp<memref::CastOp>()) {
      if (!changed) {
        rewriter.modifyOpInPlace(
            op, [&]() { op.getTargetViewMutable().assign(cast.getSource()); });
      } else {
        op.getTargetViewMutable().assign(cast.getSource());
      }
      changed = true;
    }

    return success(changed);
  }
};

struct FoldCopyToTensorType : public OpRewritePattern<CopyToTensorOp> {
  using OpRewritePattern<CopyToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyToTensorOp op,
                                PatternRewriter &rewriter) const override {
    Value buffer = op.getBuffer();
    if (auto cast = buffer.getDefiningOp<memref::CastOp>()) {
      buffer = cast.getSource();
    }

    auto sourceView = op.getSourceView();
    auto sourceType = llvm::dyn_cast<MemRefType>(sourceView.getType());
    if (!sourceType)
      return failure();

    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    bool needsBufferUpdate = (buffer != op.getBuffer());
    bool needsTypeUpdate =
        sourceType.hasStaticShape() && !resultType.hasStaticShape();

    if (!needsBufferUpdate && !needsTypeUpdate)
      return failure();

    auto newResultType = resultType;
    if (needsTypeUpdate) {
      newResultType = CopyToTensorOp::inferResultType(sourceType);
    }

    rewriter.replaceOpWithNewOp<CopyToTensorOp>(
        op, newResultType, op.getSourceView(), buffer, op.getMemoryAttr(),
        op.getProvenance(), op.getInterconnectAttr(), op.getBroadcastAttr());
    return success();
  }
};

struct FoldSemaphoreTakeType : public OpRewritePattern<SemaphoreTakeOp> {
  using OpRewritePattern<SemaphoreTakeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SemaphoreTakeOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    // Unwrap memref.cast to reach the underlying static alloc
    if (auto cast = source.getDefiningOp<memref::CastOp>())
      source = cast.getSource();

    auto sourceType = llvm::dyn_cast<MemRefType>(source.getType());
    if (!sourceType)
      return failure();

    auto resultType = llvm::dyn_cast<MemRefType>(op.getType());
    if (!resultType)
      return failure();

    bool needsSourceUpdate = (source != op.getSource());
    // Propagate static shape if source is now fully static
    bool needsTypeUpdate =
        sourceType.hasStaticShape() && !resultType.hasStaticShape();

    if (!needsSourceUpdate && !needsTypeUpdate)
      return failure();

    auto newResultType = needsTypeUpdate ? sourceType : resultType;
    rewriter.replaceOpWithNewOp<SemaphoreTakeOp>(op, newResultType, source);
    return success();
  }
};

struct DropUnusedSemaphoreTakeGive : public OpRewritePattern<SemaphoreTakeOp> {
  using OpRewritePattern<SemaphoreTakeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SemaphoreTakeOp op,
                                PatternRewriter &rewriter) const override {
    Value semaphore = op.getResult();
    if (!semaphore.hasOneUse())
      return failure();

    auto giveOp = dyn_cast<SemaphoreGiveOp>(*semaphore.user_begin());
    if (!giveOp || giveOp.getSource() != semaphore)
      return failure();

    rewriter.eraseOp(giveOp);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Staticize loom.broadcast by peeking through tensor.cast on ins/init and
/// unwrapping casts on ins/init. Broadcast may intentionally have init/outs
/// type different from result type, so we never derive result type from init.
/// Instead, if users apply tensor.cast to a more-static ranked tensor type,
/// migrate that cast target type onto broadcast's result directly.
struct StaticizeBroadcast : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern<BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    Value ins = op.getIns();
    if (auto cast = ins.getDefiningOp<tensor::CastOp>())
      ins = cast.getSource();

    Value init = op.getInit();
    if (auto cast = init.getDefiningOp<tensor::CastOp>())
      init = cast.getSource();

    if (op->getNumResults() == 0)
      return failure();

    auto resultType = llvm::dyn_cast<RankedTensorType>(op->getResultTypes()[0]);
    if (!resultType)
      return failure();

    RankedTensorType migratedResultType = resultType;
    SmallVector<tensor::CastOp> foldableCasts;

    Value opResult = op->getResult(0);
    for (Operation *user : opResult.getUsers()) {
      auto castOp = dyn_cast<tensor::CastOp>(user);
      if (!castOp || castOp.getSource() != opResult)
        continue;
      auto castType = llvm::dyn_cast<RankedTensorType>(castOp.getType());
      if (!castType)
        continue;
      if (castType.hasStaticShape()) {
        if (migratedResultType == resultType || migratedResultType == castType) {
          migratedResultType = castType;
          foldableCasts.push_back(castOp);
        }
      }
    }

    bool needsInsUpdate = (ins != op.getIns());
    bool needsInitUpdate = (init != op.getInit());
    bool needsTypeUpdate = (migratedResultType != resultType);

    if (!needsInsUpdate && !needsInitUpdate && !needsTypeUpdate)
      return failure();

    auto newOp = rewriter.create<BroadcastOp>(
        op.getLoc(), TypeRange{migratedResultType}, ins, init, op.getDimAttr());
    Value newResult = newOp->getResult(0);

    if (needsTypeUpdate) {
      for (auto castOp : foldableCasts) {
        if (castOp->getBlock())
          rewriter.replaceOp(castOp, newResult);
      }
      if (!opResult.use_empty()) {
        auto bridgeCast = rewriter.create<tensor::CastOp>(
            op.getLoc(), resultType, newResult);
        rewriter.replaceOp(op, bridgeCast.getResult());
      } else {
        rewriter.eraseOp(op);
      }
    } else {
      rewriter.replaceOp(op, newOp.getResults());
    }
    return success();
  }
};

struct StaticizeReduceSum : public OpRewritePattern<ReduceSumOp> {
  using OpRewritePattern<ReduceSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceSumOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    if (auto cast = input.getDefiningOp<tensor::CastOp>())
      input = cast.getSource();

    Value init = op.getInit();
    if (auto cast = init.getDefiningOp<tensor::CastOp>())
      init = cast.getSource();

    // Only canonicalize tensor-mode ops (those with a result).
    if (op->getNumResults() == 0)
      return failure();

    auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return failure();

    auto resultType = llvm::dyn_cast<RankedTensorType>(op->getResultTypes()[0]);
    if (!resultType)
      return failure();

    bool needsInputUpdate = (input != op.getInput());
    bool needsInitUpdate = (init != op.getInit());
    bool needsTypeUpdate =
        inputType.hasStaticShape() && !resultType.hasStaticShape();

    if (!needsInputUpdate && !needsInitUpdate && !needsTypeUpdate)
      return failure();

    Type newResultType = needsTypeUpdate ? inputType : resultType;

    auto newOp = rewriter.create<ReduceSumOp>(
        op.getLoc(), newResultType, input, init, op.getUlX(), op.getUlY(),
        op.getLrX(), op.getLrY());

    if (newResultType != resultType) {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                  newOp.getResult());
    } else {
      rewriter.replaceOp(op, newOp.getResults());
    }
    return success();
  }
};

struct StaticizeCopy : public OpRewritePattern<CopyOp> {
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    if (auto cast = source.getDefiningOp<memref::CastOp>()) {
      source = cast.getSource();
    }

    Value destination = op.getDestination();
    if (auto cast = destination.getDefiningOp<memref::CastOp>()) {
      destination = cast.getSource();
    }

    if (source == op.getSource() && destination == op.getDestination()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<CopyOp>(
        op, source, destination, op.getSrcMemSpaceAttr(),
        op.getDstMemSpaceAttr(), op.getArea(), op.getStaticAreaAttr(),
        op.getUlX(), op.getUlY(), op.getLrX(), op.getLrY(),
        op.getReclaimAttr());
    return success();
  }
};

struct StaticizeGather : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const override {
    Value across = op.getAcross();
    bool dropAcross = false;
    if (across) {
      IntegerAttr constAttr;
      if (matchPattern(across, m_Constant(&constAttr)))
        dropAcross = true;
    }

    Value source = op.getSource();
    if (auto cast = source.getDefiningOp<memref::CastOp>())
      source = cast.getSource();

    Value destination = op.getDestination();
    if (auto cast = destination.getDefiningOp<memref::CastOp>())
      destination = cast.getSource();

    if (source == op.getSource() && destination == op.getDestination() &&
        !dropAcross)
      return failure();

    rewriter.replaceOpWithNewOp<GatherOp>(
        op, source, destination, op.getSrcMemSpaceAttr(),
        op.getDstMemSpaceAttr(), dropAcross ? Value{} : across, op.getArea(),
        op.getStaticAreaAttr(), op.getUlX(), op.getUlY(), op.getLrX(),
        op.getLrY());
    return success();
  }
};

} // namespace

void SubviewOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<FoldSubviewConstants>(context);
}

void AllocOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<FoldAllocConstants>(context);
}

void InitTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<FoldInitTensorConstants, StaticizeScfFor>(context);
}

void CopyFromTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<StaticizeCopyFromTensor>(context);
}

void CopyToTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<FoldCopyToTensorType>(context);
}

void SemaphoreTakeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<FoldSemaphoreTakeType, DropUnusedSemaphoreTakeGive>(context);
}

void CopyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<StaticizeCopy>(context);
}

void ReduceSumOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<StaticizeReduceSum>(context);
}

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<StaticizeGather>(context);
}

void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<StaticizeBroadcast>(context);
}

//===----------------------------------------------------------------------===//
// BufferizeToTensorOp / BufferizeToMemrefOp
//===----------------------------------------------------------------------===//

::mlir::SmallVector<::mlir::Value>
loom::BufferizeToTensorOp::getHandoffTargetTensors() {
  return {getResult()};
}

::mlir::SmallVector<::mlir::Value>
loom::BufferizeToMemrefOp::getHandoffTargetTensors() {
  return {getSource()};
}

::mlir::RankedTensorType
loom::BufferizeToTensorOp::inferResultType(
    ::llvm::ArrayRef<int64_t> staticSizes, ::mlir::Type elementType) {
  return RankedTensorType::get(staticSizes, elementType);
}

::mlir::MemRefType loom::BufferizeToMemrefOp::inferResultType(
    ::llvm::ArrayRef<int64_t> staticSizes, ::mlir::Type elementType) {
  return MemRefType::get(staticSizes, elementType);
}

namespace {

/// Fold constant size operands of loom.bufferize_to_tensor and update the
/// result type to a static shape when all sizes become known.
struct FoldBufferizeToTensorConstants
    : public OpRewritePattern<BufferizeToTensorOp> {
  using OpRewritePattern<BufferizeToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferizeToTensorOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult, 4> sizes = op.getMixedSizes();
    bool constantsFolded = foldConstantDimensions(sizes);

    SmallVector<Value, 4> dynamicSizes;
    SmallVector<int64_t, 4> staticSizes;
    dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);

    bool allStatic = llvm::all_of(
        staticSizes, [](int64_t s) { return s != ShapedType::kDynamic; });

    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    bool needsTypeUpdate = allStatic && !resultType.hasStaticShape();

    if (!constantsFolded && !needsTypeUpdate)
      return failure();

    Type newResultType = resultType;
    if (needsTypeUpdate) {
      newResultType = BufferizeToTensorOp::inferResultType(
          staticSizes, resultType.getElementType());
    }

    auto newOp = BufferizeToTensorOp::create(
        rewriter, op.getLoc(), newResultType, op.getSource(), dynamicSizes,
        rewriter.getDenseI64ArrayAttr(staticSizes));

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// If the source tensor of loom.bufferize_to_memref has a fully static shape
/// but the result memref is dynamic, update the result type to static.
struct FoldBufferizeToMemrefType
    : public OpRewritePattern<BufferizeToMemrefOp> {
  using OpRewritePattern<BufferizeToMemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferizeToMemrefOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSource();
    if (auto castOp = src.getDefiningOp<tensor::CastOp>()) {
      src = castOp.getSource();
    }
    auto srcType = llvm::dyn_cast<RankedTensorType>(src.getType());
    if (!srcType || !srcType.hasStaticShape())
      return failure();

    auto resultType = llvm::dyn_cast<MemRefType>(op.getType());
    if (!resultType)
      return failure();

    auto newResultType = BufferizeToMemrefOp::inferResultType(
        srcType.getShape(), srcType.getElementType());

    bool needsTypeUpdate = (newResultType != resultType);
    bool needsSourceUpdate = (src != op.getSource());

    if (!needsTypeUpdate && !needsSourceUpdate)
      return failure();

    auto newOp = BufferizeToMemrefOp::create(rewriter, op.getLoc(),
                                             newResultType, src);
    if (needsTypeUpdate) {
      rewriter.replaceOpWithNewOp<memref::CastOp>(op, resultType,
                                                  newOp.getResult());
    } else {
      rewriter.replaceOp(op, newOp.getResult());
    }
    return success();
  }
};

} // namespace

void BufferizeToTensorOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<FoldBufferizeToTensorConstants>(context);
}

void BufferizeToMemrefOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<FoldBufferizeToMemrefType>(context);
}
