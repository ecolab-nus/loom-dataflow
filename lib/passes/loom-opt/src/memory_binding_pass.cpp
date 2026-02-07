#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Loom dialect headers
#include "LoomDialect.h.inc"
#include "mlir/Interfaces/ViewLikeInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

/// Pattern 1: Lower memref.subview + bufferization.to_tensor
/// to loom.subview + loom.alloc + loom.copy_to_tensor
struct ReadBlockLoadingLowering
    : public OpRewritePattern<bufferization::ToTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto subviewOp = op.getBuffer().getDefiningOp<memref::SubViewOp>();
    if (!subviewOp)
      return failure();

    Location loc = op.getLoc();

    // 1. Create loom.subview
    auto subviewResultType = loom::SubviewOp::inferResultType(
        cast<MemRefType>(subviewOp.getSource().getType()),
        subviewOp.getStaticOffsets(), subviewOp.getStaticSizes(),
        subviewOp.getStaticStrides());
    auto loomSubviewOp = loom::SubviewOp::create(
        rewriter, loc, subviewResultType, subviewOp.getSource(),
        subviewOp.getOffsets(), subviewOp.getSizes(), subviewOp.getStrides(),
        subviewOp.getStaticOffsets(), subviewOp.getStaticSizes(),
        subviewOp.getStaticStrides(), false, false, false);

    // 2. Create loom.alloc on @L1
    auto subviewType = cast<MemRefType>(subviewOp.getResult().getType());
    auto allocResultType =
        MemRefType::get(subviewType.getShape(), subviewType.getElementType());
    auto allocOp = loom::AllocOp::create(
        rewriter, loc, allocResultType, subviewOp.getSizes(),
        subviewOp.getStaticSizesAttr(), nullptr, rewriter.getI64IntegerAttr(1),
        SymbolRefAttr::get(rewriter.getContext(), "L1"));

    // 3. Create loom.copy_to_tensor
    auto emptyArray = rewriter.getArrayAttr({});
    auto defaultBroadcast = rewriter.getI64ArrayAttr({1, 1});
    rewriter.replaceOp(op, loom::CopyToTensorOp::create(
                               rewriter, loc, op.getType(),
                               loomSubviewOp.getResult(), allocOp.getResult(),
                               nullptr, Value(), emptyArray, defaultBroadcast));

    // We can't safely remove subview yet if it has other uses,
    // but usually it's just used by to_tensor in this pattern.
    if (subviewOp->use_empty()) {
      rewriter.eraseOp(subviewOp);
    }

    return success();
  }
};

/// Pattern 2: Transform tensor.empty (source of iter_args chain)
/// to loom.alloc + loom.init_tensor
struct OutputTensorInitLowering : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const override {
    // We need to check if this leads to an affine.for iter_args via linalg.fill
    bool leadsToIterArgs = false;
    for (auto &use : op.getResult().getUses()) {
      if (auto fillOp = dyn_cast<linalg::FillOp>(use.getOwner())) {
        for (auto &fillUse : fillOp.getResult(0).getUses()) {
          if (auto forOp = dyn_cast<affine::AffineForOp>(fillUse.getOwner())) {
            // Check if it's used as an initial value for iter_args
            for (auto initVal : forOp.getInits()) {
              if (initVal == fillOp.getResult(0)) {
                leadsToIterArgs = true;
                break;
              }
            }
          }
        }
      }
    }

    if (!leadsToIterArgs)
      return failure();

    Location loc = op.getLoc();
    // 1. Create loom.alloc on @L1
    auto tensorType = op.getType();
    auto allocResultType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    auto allocOp = loom::AllocOp::create(
        rewriter, loc, allocResultType, op.getDynamicSizes(),
        rewriter.getDenseI64ArrayAttr(op.getType().getShape()), nullptr,
        rewriter.getI64IntegerAttr(1),
        SymbolRefAttr::get(rewriter.getContext(), "L1"));

    // 2. Create loom.init_tensor
    rewriter.replaceOp(
        op, loom::InitTensorOp::create(
                rewriter, loc, op.getType(), allocOp.getResult(),
                op.getDynamicSizes(),
                rewriter.getDenseI64ArrayAttr(op.getType().getShape())));

    return success();
  }
};

/// Pattern 3: Transform write-back chain
/// (memref.subview + bufferization.to_buffer + memref.copy)
/// to loom.subview + loom.copy_from_tensor
struct WriteBackLowering : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto toBufferOp = op.getSource().getDefiningOp<bufferization::ToBufferOp>();
    auto subviewOp = op.getTarget().getDefiningOp<memref::SubViewOp>();

    if (!toBufferOp || !subviewOp)
      return failure();

    Location loc = subviewOp.getLoc();

    // 1. Create loom.subview
    auto subviewResultType = loom::SubviewOp::inferResultType(
        cast<MemRefType>(subviewOp.getSource().getType()),
        subviewOp.getStaticOffsets(), subviewOp.getStaticSizes(),
        subviewOp.getStaticStrides());
    auto loomSubviewOp = loom::SubviewOp::create(
        rewriter, loc, subviewResultType, subviewOp.getSource(),
        subviewOp.getOffsets(), subviewOp.getSizes(), subviewOp.getStrides(),
        subviewOp.getStaticOffsets(), subviewOp.getStaticSizes(),
        subviewOp.getStaticStrides(), false, false, false);

    // 2. Create loom.copy_from_tensor
    loom::CopyFromTensorOp::create(rewriter, loc, toBufferOp.getTensor(),
                                   loomSubviewOp.getResult(), nullptr);

    // Erase original ops
    rewriter.eraseOp(op);
    if (toBufferOp->use_empty())
      rewriter.eraseOp(toBufferOp);
    if (subviewOp->use_empty())
      rewriter.eraseOp(subviewOp);

    return success();
  }
};

struct MemoryBindingPass
    : public PassWrapper<MemoryBindingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryBindingPass)

  StringRef getArgument() const override { return "loom-memory-binding"; }

  StringRef getDescription() const override {
    return "Bind memory allocations to tensor operations in loom framework";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect, memref::MemRefDialect,
                    tensor::TensorDialect, linalg::LinalgDialect,
                    bufferization::BufferizationDialect, loom::LoomDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReadBlockLoadingLowering, OutputTensorInitLowering,
                 WriteBackLowering>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createMemoryBindingPass() {
  return std::make_unique<MemoryBindingPass>();
}
