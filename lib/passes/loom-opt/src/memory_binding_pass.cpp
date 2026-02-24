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
    applyPatternRewrites();
  }

private:
  /// Trace the outs operand of a linalg op back to its defining loom.alloc.
  /// After applyPatternRewrites, all bufferization.to_tensor ops have been
  /// converted to loom.copy_to_tensor, so only loom ops remain.
  Value traceOutsToAlloc(Value tensor) {
    // 1. Primary path: Use the pre-computed allocation plan
    auto it = plan.tensorToBufferMap.find(tensor);
    if (it != plan.tensorToBufferMap.end()) {
      auto allocIt =
          colorToAlloc.find({it->second.bucketKey, it->second.colorId});
      if (allocIt != colorToAlloc.end())
        return allocIt->second;
    }

    // 2. Secondary path: Manual trace (for intermediate or untracked tensors)
    Value current = tensor;
    while (current) {
      if (auto initOp = current.getDefiningOp<loom::InitTensorOp>())
        return initOp.getBuffer();
      if (auto copyOp = current.getDefiningOp<loom::CopyToTensorOp>())
        return copyOp.getBuffer();
      if (auto toTensorOp = current.getDefiningOp<bufferization::ToTensorOp>())
        return toTensorOp->getOperand(0);

      if (auto castOp = current.getDefiningOp<tensor::CastOp>()) {
        current = castOp.getSource();
        continue;
      }

      if (auto linalgOp =
              dyn_cast_or_null<linalg::LinalgOp>(current.getDefiningOp())) {
        if (linalgOp.getDpsInits().size() == 1) {
          current = linalgOp.getDpsInits()[0];
          continue;
        }
      }

      // Handle loop-carried dependencies (iter_args)
      if (auto blockArg = mlir::dyn_cast<BlockArgument>(current)) {
        if (auto forOp = dyn_cast<affine::AffineForOp>(
                blockArg.getOwner()->getParentOp())) {
          unsigned argIdx = blockArg.getArgNumber() -
                            forOp.getBody()->getNumArguments() +
                            forOp.getNumIterOperands();
          current = forOp.getInits()[argIdx];
          continue;
        }
      }
      break;
    }
    return nullptr;
  }

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
    // Pass 1: Replace tensor.empty outs for ALL members across ALL VBs.
    // This must happen first so traceOutsToAlloc can resolve allocs accurately
    // in Pass 2.
    for (const auto &[sig, bucket] : analysisCtx.getBuckets()) {
      for (const auto &vb : bucket.virtualBuffers) {
        if (vb->members.empty())
          continue;

        Value allocVal = colorToAlloc.lookup({sig, vb->color});
        if (!allocVal)
          continue;

        // For Fused/LoopCarried VBs, find the iterarg (BlockArgument) member.
        Value iterArg = nullptr;
        Operation *forOp = nullptr;
        if (vb->type == VBType::Fused || vb->type == VBType::LoopCarried) {
          for (TensorNode *member : vb->members) {
            if (auto blockArg = mlir::dyn_cast<BlockArgument>(member->value)) {
              iterArg = blockArg;
              forOp = blockArg.getOwner()->getParentOp();
              break;
            }
          }
        }

        for (TensorNode *member : vb->members) {
          auto linalgOp =
              dyn_cast_or_null<linalg::LinalgOp>(member->definingOp);
          if (!linalgOp)
            continue;

          for (OpOperand &outsOp : linalgOp.getDpsInitsMutable()) {
            if (outsOp.get().getDefiningOp<tensor::EmptyOp>()) {
              // If this VB has an iterarg AND the op is inside the loop,
              // use iterarg as the outs to preserve SSA chain.
              if (iterArg && forOp->isProperAncestor(linalgOp)) {
                outsOp.set(iterArg);
              } else {
                outsOp.set(getOrCreateInitTensor(allocVal));
              }
            }
          }
        }
      }
    }

    // Pass 2: Handle iterarg births and non-iterarg birthNode comparisons.
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

        // Step 1: IterArg handling (special case)
        if (auto blockArg = mlir::dyn_cast<BlockArgument>(birthNode->value)) {
          // IterArg: trace to init value, clone its defining op onto PB
          auto forOp =
              dyn_cast<affine::AffineForOp>(blockArg.getOwner()->getParentOp());
          assert(forOp && "iterarg birthNode must be inside an affine.for");
          unsigned argIdx = blockArg.getArgNumber() -
                            forOp.getBody()->getNumArguments() +
                            forOp.getNumIterOperands();
          Value initVal = forOp.getInits()[argIdx];
          Operation *initDefOp = initVal.getDefiningOp();
          if (initDefOp && isa<linalg::LinalgOp>(initDefOp)) {
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
          }
        } else {
          // Step 2: Non-iterarg birth. Redirect defining LinalgOp if exists.
          Operation *defOp = birthNode->value.getDefiningOp();
          if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(defOp)) {
            OpBuilder builder(context);
            builder.setInsertionPoint(linalgOp);

            // We assume single output for now as per Loom patterns
            OpOperand &outsOperand = linalgOp.getDpsInitsMutable()[0];
            Value outsTensor = outsOperand.get();
            Value currentAlloc = traceOutsToAlloc(outsTensor);

            if (currentAlloc && currentAlloc != allocVal) {
              Value newOuts = getOrCreateInitTensor(allocVal);
              auto copyOp = builder.create<linalg::CopyOp>(linalgOp.getLoc(),
                                                           outsTensor, newOuts);
              outsOperand.set(copyOp.getResult(0));
              // Update result type to match new init_tensor
              linalgOp->getResult(0).setType(newOuts.getType());
            } else if (!currentAlloc) {
              // No known source buffer (e.g. tensor.empty or external subview)
              outsOperand.set(getOrCreateInitTensor(allocVal));
              linalgOp->getResult(0).setType(outsOperand.get().getType());
            }
          }
        }
      }
    }

    // Pass 3: Insert yield copy-backs for split yields.
    for (const auto &split : analysisCtx.getSplitYields()) {
      auto itArg = plan.tensorToBufferMap.find(split.iterArg);
      auto itYield = plan.tensorToBufferMap.find(split.yieldVal);

      if (itArg == plan.tensorToBufferMap.end() ||
          itYield == plan.tensorToBufferMap.end())
        continue;

      // If they are on different physical buffers, we need a copy
      if (itArg->second.colorId != itYield->second.colorId ||
          itArg->second.bucketKey != itYield->second.bucketKey) {

        Value yieldVal = split.yieldVal;

        // Find the affine.yield op
        auto forOp = cast<affine::AffineForOp>(
            mlir::cast<BlockArgument>(split.iterArg).getOwner()->getParentOp());
        auto yieldOp =
            cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator());

        OpBuilder builder(context);
        builder.setInsertionPoint(yieldOp);

        auto copyOp = builder.create<linalg::CopyOp>(yieldOp.getLoc(), yieldVal,
                                                     itArg->first);

        // Update the yield operand
        yieldOp.setOperand(split.iterArgIndex, copyOp.getResult(0));
      }
    }
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

private:
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
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createMemoryBindingPass() {
  return std::make_unique<MemoryBindingPass>();
}
