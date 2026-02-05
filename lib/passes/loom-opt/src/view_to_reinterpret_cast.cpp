#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

/// Helper function to extract strides from a MemRefType.
SmallVector<int64_t, 4> getSourceStrides(MemRefType type) {
  auto layout = type.getLayout();
  if (auto stridedLayout = llvm::dyn_cast<StridedLayoutAttr>(layout)) {
    return llvm::to_vector<4>(stridedLayout.getStrides());
  }
  // Fallback to row-major
  SmallVector<int64_t, 4> strides;
  int64_t stride = 1;
  strides.resize(type.getRank());
  for (int i = type.getRank() - 1; i >= 0; --i) {
    strides[i] = stride;
    if (type.getDimSize(i) != ShapedType::kDynamic) {
      stride *= type.getDimSize(i);
    } else {
      stride = ShapedType::kDynamic;
    }
  }
  return strides;
}

class SubviewToReinterpretCastPass
    : public PassWrapper<SubviewToReinterpretCastPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SubviewToReinterpretCastPass)

  StringRef getArgument() const override {
    return "loom-subview-to-reinterpret-cast";
  }

  StringRef getDescription() const override {
    return "Convert loom.subview to loom.reinterpret_cast using affine mapping "
           "for linearized offset";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    module.walk([&](loom::SubviewOp subviewOp) {
      builder.setInsertionPoint(subviewOp);
      Location loc = subviewOp.getLoc();

      MemRefType sourceType = subviewOp.getSourceType();
      SmallVector<int64_t, 4> sourceStrides = getSourceStrides(sourceType);

      // 1. Compute linearized offset via affine.apply
      SmallVector<OpFoldResult, 4> viewOffsets = subviewOp.getMixedOffsets();

      // We'll create an affine map: (d0, d1, ...) -> (d0 * S0 + d1 * S1 + ...)
      // where Si are the source strides.
      SmallVector<AffineExpr, 4> exprs;
      for (size_t i = 0; i < sourceStrides.size(); ++i) {
        exprs.push_back(builder.getAffineDimExpr(i) * sourceStrides[i]);
      }

      AffineExpr sumExpr = builder.getAffineConstantExpr(0);
      for (auto expr : exprs) {
        sumExpr = sumExpr + expr;
      }

      AffineMap offsetMap = AffineMap::get(sourceStrides.size(), 0, sumExpr);

      // Extract dynamic values for the map
      SmallVector<Value, 4> mapOperands;
      for (auto ofr : viewOffsets) {
        if (auto value = ofr.dyn_cast<Value>()) {
          mapOperands.push_back(value);
        } else {
          // Constant offset
          int64_t val = cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
          mapOperands.push_back(
              builder.create<arith::ConstantIndexOp>(loc, val));
        }
      }

      Value linearizedOffset =
          builder.create<affine::AffineApplyOp>(loc, offsetMap, mapOperands);

      // 2. Compute result strides: Strides_rc[i] = Strides_view[i] *
      // Strides_org[i]
      SmallVector<OpFoldResult, 4> viewStrides = subviewOp.getMixedStrides();
      SmallVector<OpFoldResult, 4> rcStrides;
      for (size_t i = 0; i < viewStrides.size(); ++i) {
        if (auto attr = viewStrides[i].dyn_cast<Attribute>()) {
          int64_t viewStride = cast<IntegerAttr>(attr).getInt();
          if (viewStride != ShapedType::kDynamic &&
              sourceStrides[i] != ShapedType::kDynamic) {
            rcStrides.push_back(
                builder.getIndexAttr(viewStride * sourceStrides[i]));
          } else {
            rcStrides.push_back(builder.getIndexAttr(ShapedType::kDynamic));
          }
        } else {
          rcStrides.push_back(viewStrides[i]);
        }
      }

      // 3. Create loom.reinterpret_cast
      MemRefType resultType = cast<MemRefType>(subviewOp.getResult().getType());

      SmallVector<Value, 4> dynamicRcSizes;
      SmallVector<int64_t, 4> staticRcSizes;
      dispatchIndexOpFoldResults(subviewOp.getMixedSizes(), dynamicRcSizes,
                                 staticRcSizes);

      SmallVector<Value, 4> dynamicRcStrides;
      SmallVector<int64_t, 4> staticRcStrides;
      dispatchIndexOpFoldResults(rcStrides, dynamicRcStrides, staticRcStrides);

      auto rcOp = builder.create<loom::ReinterpretCastOp>(
          loc, resultType, subviewOp.getSource(), ValueRange{linearizedOffset},
          dynamicRcSizes, dynamicRcStrides,
          builder.getDenseI64ArrayAttr({ShapedType::kDynamic}),
          builder.getDenseI64ArrayAttr(staticRcSizes),
          builder.getDenseI64ArrayAttr(staticRcStrides));

      subviewOp.getResult().replaceAllUsesWith(rcOp.getResult());
      subviewOp.erase();
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createSubviewToReinterpretCastPass() {
  return std::make_unique<SubviewToReinterpretCastPass>();
}
