//===- LoomOps.cpp - LOOM Dialect Operations -----------------------------===//
//
// Implementation of the LOOM operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace loom;

#include "LoomDialect.cpp.inc"

#include "LoomEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.cpp.inc"

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

LogicalResult loom::ConstraintSpaceOp::verify() {
  llvm::DenseMap<StringAttr, Location> variableNames;
  for (Operation &op : getBodyBlock()->getOperations()) {
    if (auto symbolicVar = dyn_cast<loom::SymbolicVarOp>(&op)) {
      StringAttr varName = symbolicVar.getNameAttr();
      auto [it, inserted] =
          variableNames.try_emplace(varName, symbolicVar.getLoc());
      if (!inserted) {
        return symbolicVar.emitOpError("duplicate symbolic variable name '")
               << varName.getValue() << "' in constraint space; "
               << "previously defined at " << it->second;
      }
    }
  }
  return success();
}

LogicalResult loom::GetSymbolicBlockSizeOp::verify() {
  SymbolRefAttr symbolRef = getSymbolRef();
  if (symbolRef.getNestedReferences().size() != 1) {
    return emitOpError("symbol reference must have format @space::@var, got ")
           << symbolRef;
  }
  return success();
}

LogicalResult loom::ExpressionOp::verify() {
  auto operands = getOperands();
  auto coeffs = getCoeffs();
  auto logic = getLogic();
  if (logic == "add") {
    if (operands.size() != coeffs.size()) {
      return emitOpError("number of operands must match number of coefficients "
                         "for 'add' logic");
    }
  } else if (logic == "mul") {
    if (operands.size() != 2) {
      return emitOpError("multiplication must have exactly two operands");
    }
    if (coeffs.size() != 2) {
      return emitOpError(
          "multiplication must have two coefficients (typically {1, 1})");
    }
  } else {
    return emitOpError("unsupported logic type: ") << logic;
  }
  return success();
}

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

    // Offset is 0 for initial default layout
    resultOffset = 0;
    for (size_t i = 0; i < staticOffsets.size(); ++i) {
      // Recompute original stride for offset calculation
      int64_t sourceStride = 1;
      for (size_t j = i + 1; j < (size_t)sourceType.getRank(); ++j) {
        if (sourceType.getDimSize(j) == ShapedType::kDynamic) {
          sourceStride = ShapedType::kDynamic;
          break;
        }
        sourceStride *= sourceType.getDimSize(j);
      }

      if (staticOffsets[i] != ShapedType::kDynamic &&
          sourceStride != ShapedType::kDynamic &&
          resultOffset != ShapedType::kDynamic) {
        resultOffset += staticOffsets[i] * sourceStride;
      } else {
        resultOffset = ShapedType::kDynamic;
      }
    }
  }

  auto layout = StridedLayoutAttr::get(sourceType.getContext(), resultOffset,
                                       resultStrides);

  return MemRefType::get(resultShape, elementType, layout, memorySpace);
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

    rewriter.replaceOpWithNewOp<SubviewOp>(
        op,
        SubviewOp::inferResultType(op.getSourceType(), staticOffsets,
                                   staticSizes, staticStrides),
        op.getSource(), dynamicOffsets, dynamicSizes, dynamicStrides,
        rewriter.getDenseI64ArrayAttr(staticOffsets),
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

/// Pattern to update affine.for iter_args and results when init args are
/// staticized.
struct StaticizeAffineFor : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp op,
                                PatternRewriter &rewriter) const override {
    bool needsUpdate = false;
    SmallVector<Type> newIterTypes;
    SmallVector<Value> newInits;

    for (auto [initArg, result] : llvm::zip(op.getInits(), op.getResults())) {
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

    // Create new for loop with updated types
    int64_t step = op.getStep().getSExtValue();
    auto newForOp = rewriter.create<affine::AffineForOp>(
        op.getLoc(), op.getLowerBoundOperands(), op.getLowerBoundMap(),
        op.getUpperBoundOperands(), op.getUpperBoundMap(), step, newInits);

    // Move the body of the old loop to the new one
    newForOp.getRegion().takeBody(op.getRegion());

    // Update block argument types for the iteration arguments
    for (auto [idx, blockArg] : llvm::enumerate(newForOp.getRegionIterArgs())) {
      blockArg.setType(newIterTypes[idx]);
    }

    // Update the yield inside the loop to use the static types
    auto yieldOp =
        cast<affine::AffineYieldOp>(newForOp.getBody()->getTerminator());
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
          auto castOp = mlir::tensor::CastOp::create(
              rewriter, yieldOp.getLoc(), newIterTypes[idx], yieldVal);
          newYieldOperands.push_back(castOp.getResult());
        }
      } else {
        newYieldOperands.push_back(yieldVal);
      }
    }
    rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(yieldOp,
                                                       newYieldOperands);

    // Replace old results with casts from new results
    rewriter.setInsertionPointAfter(newForOp);
    SmallVector<Value> replacedResults;
    for (auto [idx, result] : llvm::enumerate(newForOp.getResults())) {
      if (result.getType() != op.getResult(idx).getType()) {
        auto castOp = mlir::tensor::CastOp::create(
            rewriter, op.getLoc(), op.getResult(idx).getType(), result);
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
  results.add<FoldInitTensorConstants, StaticizeAffineFor>(context);
}

void CopyFromTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<StaticizeCopyFromTensor>(context);
}

void CopyToTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<FoldCopyToTensorType>(context);
}
