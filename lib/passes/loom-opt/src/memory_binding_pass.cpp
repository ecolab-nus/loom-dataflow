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
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <chrono>
#include <fstream>

// Loom dialect and analysis headers
#include "LoomDialect.h.inc"
#include "static_memory_analyser.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace loom;

namespace {

/// Pattern 1: Lower memref.subview + bufferization.to_tensor
/// to loom.subview + loom.alloc + loom.copy_to_tensor
struct ReadBlockLoadingLowering
    : public OpRewritePattern<bufferization::ToTensorOp> {
  const llvm::DenseMap<std::pair<ShapeSignature, int>, Value> &colorToAlloc;
  const llvm::DenseMap<Value, LoomAllocationPlan::Assignment>
      &tensorToBufferMap;

  ReadBlockLoadingLowering(
      MLIRContext *ctx,
      const llvm::DenseMap<std::pair<ShapeSignature, int>, Value> &c2a,
      const llvm::DenseMap<Value, LoomAllocationPlan::Assignment> &t2b)
      : OpRewritePattern(ctx), colorToAlloc(c2a), tensorToBufferMap(t2b) {}

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

    // 2. Look up the alloc from the coloring plan
    auto it = tensorToBufferMap.find(op.getResult());
    if (it == tensorToBufferMap.end())
      return failure();
    auto allocIt =
        colorToAlloc.find({it->second.bucketKey, it->second.colorId});
    if (allocIt == colorToAlloc.end())
      return failure();
    Value allocVal = allocIt->second;

    // 3. Create loom.copy_to_tensor
    auto emptyArray = rewriter.getArrayAttr({});
    auto defaultBroadcast = rewriter.getI64ArrayAttr({1, 1});
    rewriter.replaceOp(op, loom::CopyToTensorOp::create(
                               rewriter, loc, op.getType(),
                               loomSubviewOp.getResult(), allocVal, nullptr,
                               Value(), emptyArray, defaultBroadcast));

    // We can't safely remove subview yet if it has other uses.
    if (subviewOp->use_empty()) {
      rewriter.eraseOp(subviewOp);
    }

    return success();
  }
};

/// Pattern 2: Transform write-back chain
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

/// Helper to materialize physical buffers and bind them to tensors
class PassMaterializer {
public:
  PassMaterializer(MLIRContext *ctx, func::FuncOp funcOp)
      : context(ctx), funcOp(funcOp), analysisCtx(runMemoryAnalysis(funcOp)),
        plan(analysisCtx.getAllocationPlan()) {}

  void run() {
    resolveBucketScopes();
    materializePhysicalBuffers();
    anchorVirtualBuffers();
    reinforceIterationBoundaries();
    applyPatternRewrites();
  }

private:
  Value getOrCreateInitTensor(Value allocVal) {
    if (auto it = allocToInitTensor.find(allocVal);
        it != allocToInitTensor.end())
      return it->second;

    OpBuilder b(context);
    b.setInsertionPointAfter(allocVal.getDefiningOp());
    auto memrefType = cast<MemRefType>(allocVal.getType());
    auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                            memrefType.getElementType());

    SmallVector<Value> dynamicSizes;
    if (auto allocOp = allocVal.getDefiningOp<loom::AllocOp>())
      dynamicSizes = allocOp.getSizes();

    auto initOp = loom::InitTensorOp::create(
        b, allocVal.getLoc(), tensorType, allocVal, dynamicSizes,
        b.getDenseI64ArrayAttr(memrefType.getShape()));
    allocToInitTensor[allocVal] = initOp.getResult();
    return initOp.getResult();
  }

  void resolveBucketScopes() {
    for (auto &[sig, bucket] : analysisCtx.getBucketsMutable()) {
      if (bucket.nodes.empty())
        continue;

      Operation *op = bucket.nodes.front().definingOp;
      while (op && !isa<affine::AffineParallelOp>(op))
        op = op->getParentOp();
      bucket.scopeOp = op;
    }
  }

  void materializePhysicalBuffers() {
    for (const auto &[sig, bucket] : analysisCtx.getBuckets()) {
      if (!bucket.scopeOp)
        continue;

      OpBuilder builder(context);
      builder.setInsertionPointToStart(&bucket.scopeOp->getRegion(0).front());
      int numColors = plan.colorCountPerBucket.lookup(sig);

      for (int c = 0; c < numColors; ++c) {
        SmallVector<Value> dynamicSizes;
        SmallVector<int64_t> staticSizes;
        for (const auto &dim : sig.dims) {
          if (auto attr = mlir::dyn_cast<Attribute>(dim)) {
            staticSizes.push_back(cast<IntegerAttr>(attr).getInt());
          } else {
            staticSizes.push_back(ShapedType::kDynamic);
            dynamicSizes.push_back(mlir::dyn_cast<Value>(dim));
          }
        }
        auto allocType = MemRefType::get(staticSizes, sig.elementType);
        auto allocOp = loom::AllocOp::create(
            builder, bucket.scopeOp->getLoc(), allocType, dynamicSizes,
            builder.getDenseI64ArrayAttr(staticSizes), nullptr,
            builder.getI64IntegerAttr(1), SymbolRefAttr::get(context, "L1"));
        colorToAlloc[{sig, c}] = allocOp.getResult();
      }
    }
  }

  void anchorVirtualBuffers() {
    for (const auto &[sig, bucket] : analysisCtx.getBuckets()) {
      for (const auto &vb : bucket.virtualBuffers) {
        if (vb->members.empty())
          continue;

        TensorNode *birthNode =
            *llvm::min_element(vb->members, [](TensorNode *a, TensorNode *b) {
              return a->linearIndex < b->linearIndex;
            });

        Value allocVal = colorToAlloc.lookup({sig, vb->color});
        if (!allocVal)
          continue;

        // Step 1: Source Anchoring (vb_to_pb)
        OpBuilder builder(context);
        if (auto blockArg = mlir::dyn_cast<BlockArgument>(birthNode->value)) {
          // IterArg: trace to init value, clone its defining op onto PB
          auto forOp = dyn_cast<affine::AffineForOp>(
              blockArg.getOwner()->getParentOp());
          assert(forOp && "iterarg birthNode must be inside an affine.for");
          unsigned argIdx = blockArg.getArgNumber() -
                            forOp.getBody()->getNumArguments() +
                            forOp.getNumIterOperands();
          Value initVal = forOp.getInits()[argIdx];
          Operation *initDefOp = initVal.getDefiningOp();
          assert(initDefOp && isa<linalg::LinalgOp>(initDefOp) &&
                 "iterarg init value must be defined by a linalg op");

          // Create init_tensor bound directly to PB alloc
          Value initTensor = getOrCreateInitTensor(allocVal);
          // Clone the init op, redirect outs to init_tensor
          OpBuilder cloneBuilder(context);
          cloneBuilder.setInsertionPointAfter(initDefOp);
          Operation *clonedOp = cloneBuilder.clone(*initDefOp);
          cast<linalg::LinalgOp>(clonedOp).getDpsInitsMutable()[0].set(
              initTensor);
          clonedOp->getResult(0).setType(initTensor.getType());

          // Redirect affine.for init to cloned op's result
          forOp.getInitsMutable()[argIdx].set(clonedOp->getResult(0));
          // #endregion
        } else {
          builder.setInsertionPointAfter(birthNode->value.getDefiningOp());
        }
        // Skip pb_anchor for iter_arg: birth is block arg, do not replace its
        // uses (would break loop SSA); Step 2 still runs to replace tensor.empty
        // outs.
        if (!mlir::isa<BlockArgument>(birthNode->value)) {
          auto pbAnchor = loom::PbAnchorOp::create(
              builder, birthNode->value.getLoc(), birthNode->value.getType(),
              birthNode->value, allocVal);
          birthNode->value.replaceAllUsesExcept(pbAnchor.getResult(), pbAnchor);
        }

        // Step 2: replace tensor.empty outs with init_tensor(alloc)
        for (TensorNode *member : vb->members) {
          auto linalgOp =
              mlir::dyn_cast_or_null<linalg::LinalgOp>(member->definingOp);
          if (!linalgOp)
            continue;

          for (OpOperand &outsOp : linalgOp.getDpsInitsMutable()) {
            if (outsOp.get().getDefiningOp<tensor::EmptyOp>()) {
              outsOp.set(getOrCreateInitTensor(allocVal));
            }
          }
        }
      }
    }
  }

  void reinforceIterationBoundaries() {
    funcOp.walk([&](affine::AffineForOp forOp) {
      OpBuilder builder(context);
      builder.setInsertionPointAfter(forOp);
      for (Value result : forOp.getResults()) {
        if (auto it = plan.tensorToBufferMap.find(result);
            it != plan.tensorToBufferMap.end()) {
          if (Value alloc = colorToAlloc.lookup(
                  {it->second.bucketKey, it->second.colorId})) {
            auto pbAnchor = loom::PbAnchorOp::create(
                builder, forOp.getLoc(), result.getType(), result, alloc);
            result.replaceAllUsesExcept(pbAnchor.getResult(), pbAnchor);
          }
        }
      }
    });
  }

  void applyPatternRewrites() {
    RewritePatternSet patterns(context);
    patterns.add<ReadBlockLoadingLowering>(context, colorToAlloc,
                                           plan.tensorToBufferMap);
    patterns.add<WriteBackLowering>(context);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("Failed to apply pattern rewrites");
    }
  }

  MLIRContext *context;
  func::FuncOp funcOp;
  MemoryAnalysisContext analysisCtx;
  const LoomAllocationPlan &plan;
  llvm::DenseMap<std::pair<ShapeSignature, int>, Value> colorToAlloc;
  llvm::DenseMap<Value, Value> allocToInitTensor;
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
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Phase 1 & 2: Materialization and Pattern-based rewrites
    module.walk(
        [&](func::FuncOp funcOp) { PassMaterializer(context, funcOp).run(); });

    // Phase 3: Clean up redundant pb_anchor ops
    module.walk([&](loom::PbAnchorOp pbAnchor) {
      Value tensor = pbAnchor.getTensor();
      if (Operation *defOp = tensor.getDefiningOp()) {
        if (defOp->getDialect()->getNamespace() == "loom" ||
            isa<linalg::FillOp>(defOp)) {
          pbAnchor.getResult().replaceAllUsesWith(tensor);
          pbAnchor.erase();
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createMemoryBindingPass() {
  return std::make_unique<MemoryBindingPass>();
}
