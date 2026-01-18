/**
 * @file loom_to_memref.cpp
 * @brief Lowering pass for loom dialect to memref dialect with static type inference.
 */

#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

/// Extract static values from constant operands, mark others as dynamic.
static void processOperands(ValueRange operands,
                             SmallVectorImpl<int64_t> &staticVals,
                             SmallVectorImpl<Value> &dynamicVals) {
  for (Value v : operands) {
    APInt value;
    if (matchPattern(v, m_ConstantInt(&value))) {
      staticVals.push_back(value.getSExtValue());
    } else {
      staticVals.push_back(ShapedType::kDynamic);
      dynamicVals.push_back(v);
    }
  }
}

/// Lower loom.reinterpret_cast to memref.reinterpret_cast with static types.
struct LoomReinterpretCastLowering
    : public OpRewritePattern<loom::ReinterpretCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(loom::ReinterpretCastOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
    SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;

    processOperands(op.getOffsets(), staticOffsets, dynamicOffsets);
    processOperands(op.getSizes(), staticSizes, dynamicSizes);
    processOperands(op.getStrides(), staticStrides, dynamicStrides);

    auto sourceType = llvm::cast<BaseMemRefType>(op.getSource().getType());
    auto elementType = sourceType.getElementType();

    int64_t layoutOffset =
        staticOffsets.empty() ? 0 : staticOffsets[0];
    auto layout =
        StridedLayoutAttr::get(op.getContext(), layoutOffset, staticStrides);
    auto resultType = MemRefType::get(staticSizes, elementType, layout);

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, resultType, op.getSource(), dynamicOffsets, dynamicSizes,
        dynamicStrides, staticOffsets, staticSizes, staticStrides);

    return success();
  }
};

/// Lower loom.copy to memref.copy.
struct LoomCopyLowering : public OpRewritePattern<loom::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(loom::CopyOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getSrc(), op.getDst());
    return success();
  }
};

class LoomToMemRefLoweringPass
    : public PassWrapper<LoomToMemRefLoweringPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoomToMemRefLoweringPass)

  StringRef getArgument() const override { return "loom-to-memref"; }

  StringRef getDescription() const override {
    return "Lower loom operations to memref with static type inference";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LoomReinterpretCastLowering, LoomCopyLowering>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createLoomToMemRefLoweringPass() {
  return std::make_unique<LoomToMemRefLoweringPass>();
}


