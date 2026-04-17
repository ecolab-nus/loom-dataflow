#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

FailureOr<SmallVector<unsigned, 4>>
computeRetainedSubviewDims(loom::SubviewOp subviewOp, MemRefType resultType) {
  ArrayRef<int64_t> staticSizes = subviewOp.getStaticSizes();
  const unsigned sourceRank = staticSizes.size();
  const unsigned resultRank = resultType.getRank();

  if (resultRank > sourceRank)
    return failure();

  if (resultRank == sourceRank) {
    SmallVector<unsigned, 4> retainedDims;
    retainedDims.reserve(sourceRank);
    for (unsigned i = 0; i < sourceRank; ++i)
      retainedDims.push_back(i);
    return retainedDims;
  }

  SmallVector<unsigned, 4> retainedDims;
  retainedDims.reserve(resultRank);

  for (unsigned i = 0; i < sourceRank; ++i) {
    if (staticSizes[i] == ShapedType::kDynamic || staticSizes[i] != 1)
      retainedDims.push_back(i);
  }

  if (retainedDims.size() > resultRank)
    return failure();

  for (unsigned i = 0; i < sourceRank && retainedDims.size() < resultRank; ++i) {
    if (staticSizes[i] != ShapedType::kDynamic && staticSizes[i] == 1)
      retainedDims.push_back(i);
  }

  if (retainedDims.size() != resultRank)
    return failure();

  llvm::sort(retainedDims);
  return retainedDims;
}

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

class BridgeToOSBPass
    : public PassWrapper<BridgeToOSBPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BridgeToOSBPass)

  StringRef getArgument() const override { return "loom-bridge-to-osb"; }

  StringRef getDescription() const override {
    return "Convert loom.subview to memref.reinterpret_cast using affine "
           "mapping "
           "for linearized offset";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    bool hadFailure = false;

    module.walk([&](loom::SubviewOp subviewOp) {
      if (hadFailure)
        return WalkResult::interrupt();

      builder.setInsertionPoint(subviewOp);
      Location loc = subviewOp.getLoc();

      MemRefType sourceType = subviewOp.getSourceType();
      MemRefType resultType = cast<MemRefType>(subviewOp.getResult().getType());
      FailureOr<SmallVector<unsigned, 4>> retainedDimsOr =
          computeRetainedSubviewDims(subviewOp, resultType);
      if (failed(retainedDimsOr)) {
        subviewOp.emitError()
            << "failed to map loom.subview rank " << sourceType.getRank()
            << " to result rank " << resultType.getRank()
            << " while bridging to memref.reinterpret_cast";
        hadFailure = true;
        return WalkResult::interrupt();
      }
      SmallVector<unsigned, 4> retainedDims = *retainedDimsOr;

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
          int64_t val = cast<IntegerAttr>(cast<Attribute>(ofr)).getInt();
          mapOperands.push_back(
              arith::ConstantIndexOp::create(builder, loc, val));
        }
      }

      Value linearizedOffset =
          affine::AffineApplyOp::create(builder, loc, offsetMap, mapOperands);

      // 2. Compute result strides: Strides_rc[i] = Strides_view[i] *
      // Strides_org[i]. Keep exactly the dimensions represented by the
      // loom.subview result type: full-rank subviews retain unit dims, while
      // genuinely rank-reduced subviews keep only the retained coordinates.
      SmallVector<OpFoldResult, 4> viewStrides = subviewOp.getMixedStrides();
      SmallVector<OpFoldResult, 4> rcStrides;
      for (unsigned i : retainedDims) {
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

      // 3. Build the sizes list using the same retained dimensions so its
      // length always matches the memref.reinterpret_cast result rank.
      SmallVector<OpFoldResult, 4> rcSizes;
      SmallVector<OpFoldResult, 4> viewSizes = subviewOp.getMixedSizes();
      for (unsigned i : retainedDims)
        rcSizes.push_back(viewSizes[i]);

      const size_t expectedRank = static_cast<size_t>(resultType.getRank());
      if (rcSizes.size() != expectedRank || rcStrides.size() != expectedRank) {
        subviewOp.emitError()
            << "bridge-to-osb computed " << rcSizes.size()
            << " size values and " << rcStrides.size()
            << " stride values for result rank " << resultType.getRank();
        hadFailure = true;
        return WalkResult::interrupt();
      }

      // 4. Create memref.reinterpret_cast

      auto rcOp = memref::ReinterpretCastOp::create(
          builder, loc, resultType, subviewOp.getSource(), linearizedOffset,
          rcSizes, rcStrides);

      subviewOp.getResult().replaceAllUsesWith(rcOp.getResult());
      subviewOp.erase();
      return WalkResult::advance();
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createBridgeToOSBPass() {
  return std::make_unique<BridgeToOSBPass>();
}
