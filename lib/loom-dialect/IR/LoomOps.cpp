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
struct FoldViewConstants : public OpRewritePattern<ViewOp> {
  using OpRewritePattern<ViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ViewOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult, 4> offsets = op.getMixedOffsets();
    SmallVector<OpFoldResult, 4> sizes = op.getMixedSizes();
    SmallVector<OpFoldResult, 4> strides = op.getMixedStrides();

    auto foldConstants = [](SmallVectorImpl<OpFoldResult> &mixedValues) {
      bool changed = false;
      for (auto &value : mixedValues) {
        if (auto ssaValue = value.dyn_cast<Value>()) {
          IntegerAttr attr;
          if (matchPattern(ssaValue, m_Constant(&attr))) {
            value = attr;
            changed = true;
          }
        }
      }
      return changed;
    };

    bool changed = false;
    changed |= foldConstants(offsets);
    changed |= foldConstants(sizes);
    changed |= foldConstants(strides);

    if (!changed)
      return failure();

    // Reconstruct the op with new mixed values
    SmallVector<Value, 4> dynamicOffsets, dynamicSizes, dynamicStrides;
    SmallVector<int64_t, 4> staticOffsets, staticSizes, staticStrides;
    dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
    dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
    dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

    rewriter.replaceOpWithNewOp<ViewOp>(
        op, op.getType(), op.getSource(), dynamicOffsets, dynamicSizes,
        dynamicStrides, rewriter.getDenseI64ArrayAttr(staticOffsets),
        rewriter.getDenseI64ArrayAttr(staticSizes),
        rewriter.getDenseI64ArrayAttr(staticStrides), op.getSequentialReuse(),
        op.getSpatialReuse(), op.getTemporalReuse());
    return success();
  }
};
} // namespace

void ViewOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<FoldViewConstants>(context);
}
