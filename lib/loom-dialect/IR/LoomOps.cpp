//===- LoomOps.cpp - LOOM Dialect Operations -----------------------------===//
//
// Implementation of the LOOM operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "llvm/ADT/TypeSwitch.h"
#define GET_TYPEDEF_CLASSES
#include "LoomEnums.h.inc"
#include "LoomTypes.h.inc"
#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.h.inc"

#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace loom;

#include "LoomDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "LoomTypes.cpp.inc"

#include "LoomEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.cpp.inc"

void LoomDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LoomOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "LoomTypes.cpp.inc"
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
// ViewOp Type Inference
//===----------------------------------------------------------------------===//

MemRefType loom::ViewOp::inferResultType(MemRefType sourceType,
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

struct FoldViewConstants : public OpRewritePattern<ViewOp> {
  using OpRewritePattern<ViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ViewOp op,
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

    rewriter.replaceOpWithNewOp<ViewOp>(
        op,
        ViewOp::inferResultType(op.getSourceType(), staticOffsets, staticSizes,
                                staticStrides),
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

    if (!foldConstantDimensions(sizes))
      return failure();

    SmallVector<Value, 4> dynamicSizes;
    SmallVector<int64_t, 4> staticSizes;
    dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);

    rewriter.replaceOpWithNewOp<AllocOp>(
        op, op.getType(), dynamicSizes,
        rewriter.getDenseI64ArrayAttr(staticSizes), op.getAlignmentAttr(),
        op.getBufferCountAttr(), op.getMemoryAttr());
    return success();
  }
};

struct FoldInitTensorConstants : public OpRewritePattern<InitTensorOp> {
  using OpRewritePattern<InitTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InitTensorOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult, 4> sizes = op.getMixedSizes();

    if (!foldConstantDimensions(sizes))
      return failure();

    SmallVector<Value, 4> dynamicSizes;
    SmallVector<int64_t, 4> staticSizes;
    dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);

    rewriter.replaceOpWithNewOp<InitTensorOp>(
        op, op.getType(), op.getBufferToken(), dynamicSizes,
        rewriter.getDenseI64ArrayAttr(staticSizes));
    return success();
  }
};
} // namespace

void ViewOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<FoldViewConstants>(context);
}

void AllocOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<FoldAllocConstants>(context);
}

void InitTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<FoldInitTensorConstants>(context);
}
