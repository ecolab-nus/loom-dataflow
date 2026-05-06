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
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

// Loom dialect and analysis headers
#include "LoomDialect.h.inc"
#include "static_memory_analyser.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace loom;

namespace {

/// Pattern 1: Lower memref.subview + loom.bufferize_to_tensor
/// to loom.subview + loom.copy (DRAM→L1) + loom.bufferize_to_tensor
struct ReadBlockLoadingLowering
    : public OpRewritePattern<loom::BufferizeToTensorOp> {
  const llvm::DenseMap<std::pair<ShapeSignature, int>, Value> &colorToAlloc;
  const llvm::DenseMap<Value, LoomAllocationPlan::Assignment>
      &tensorToBufferMap;
  const llvm::DenseMap<Value, int> &tensorToVBIdMap;
  const llvm::DenseMap<int, Value> &vbIdToSemaphoreMap;

  ReadBlockLoadingLowering(
      MLIRContext *ctx,
      const llvm::DenseMap<std::pair<ShapeSignature, int>, Value> &c2a,
      const llvm::DenseMap<Value, LoomAllocationPlan::Assignment> &t2b,
      const llvm::DenseMap<Value, int> &t2v,
      const llvm::DenseMap<int, Value> &v2s)
      : OpRewritePattern(ctx), colorToAlloc(c2a), tensorToBufferMap(t2b),
        tensorToVBIdMap(t2v), vbIdToSemaphoreMap(v2s) {}

  LogicalResult matchAndRewrite(loom::BufferizeToTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto subviewOp = op.getSource().getDefiningOp<memref::SubViewOp>();
    if (!subviewOp)
      return failure();

    // 1. Look up the alloc from the coloring plan (check preconditions before
    //    creating any ops to avoid orphaned ops on failure).
    auto it = tensorToBufferMap.find(op.getResult());
    if (it == tensorToBufferMap.end())
      return failure();
    auto allocIt =
        colorToAlloc.find({it->second.bucketKey, it->second.colorId});
    if (allocIt == colorToAlloc.end())
      return failure();
    Value allocVal = allocIt->second;

    Location loc = op.getLoc();

    // 2. Create loom.subview — reuse the upstream memref.subview's result type
    //    verbatim (it is already rank-reduced when the subview is rank-reducing).
    auto subviewResultType = cast<MemRefType>(subviewOp.getResult().getType());
    auto loomSubviewOp = loom::SubviewOp::create(
        rewriter, loc, subviewResultType, subviewOp.getSource(),
        subviewOp.getOffsets(), subviewOp.getSizes(), subviewOp.getStrides(),
        subviewOp.getStaticOffsets(), subviewOp.getStaticSizes(),
        subviewOp.getStaticStrides(), false, false, false);

    // 3. Create loom.copy_to_tensor
    // Use the pre-created semaphore for the virtual buffer
    Value semaphore;
    auto vbIt = tensorToVBIdMap.find(op.getResult());
    if (vbIt != tensorToVBIdMap.end()) {
      semaphore = vbIdToSemaphoreMap.lookup(vbIt->second);
    }

    if (!semaphore) {
      // Fallback: This should ideally not happen with centralized management
      OpBuilder semBuilder(rewriter.getContext());
      semBuilder.setInsertionPoint(op);
      semaphore = loom::SemaphoreTakeOp::create(
          semBuilder, loc, cast<MemRefType>(allocVal.getType()), allocVal);
    }

    // 3. loom.copy: physically move data DRAM→L1 into the semaphore buffer.
    auto dramSymbol = SymbolRefAttr::get(rewriter.getContext(), "mem_DRAM");
    auto l1Symbol = SymbolRefAttr::get(rewriter.getContext(), "mem_L1");
    auto defaultBroadcast = rewriter.getI64ArrayAttr({1, 1});
    loom::CopyOp::create(rewriter, loc, loomSubviewOp.getResult(), semaphore,
                         dramSymbol, l1Symbol, defaultBroadcast,
                         Value{}, Value{}, Value{}, Value{});

    rewriter.replaceOp(op, loom::BufferizeToTensorOp::create(
                               rewriter, loc, op.getType(), semaphore,
                               op.getSizes(),
                               rewriter.getDenseI64ArrayAttr(
                                   op.getStaticSizes())));

    // We can't safely remove subview yet if it has other uses.
    if (subviewOp->use_empty()) {
      rewriter.eraseOp(subviewOp);
    }

    return success();
  }
};

/// Pattern 2: Transform write-back chain
/// (memref.subview + loom.bufferize_to_memref + memref.copy)
/// to loom.subview + loom.bufferize_to_memref + loom.copy (L1→DRAM)
struct WriteBackLowering : public OpRewritePattern<memref::CopyOp> {
  WriteBackLowering(MLIRContext *context,
                    const llvm::DenseMap<Value, int> &tensorToVBIdMap,
                    const llvm::DenseMap<int, Value> &vbIdToSemaphoreMap)
      : OpRewritePattern<memref::CopyOp>(context),
        tensorToVBIdMap(tensorToVBIdMap),
        vbIdToSemaphoreMap(vbIdToSemaphoreMap) {}

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto toMemrefOp = op.getSource().getDefiningOp<loom::BufferizeToMemrefOp>();
    auto subviewOp = op.getTarget().getDefiningOp<memref::SubViewOp>();

    if (!toMemrefOp || !subviewOp)
      return failure();

    Location loc = subviewOp.getLoc();

    // 1. Create loom.subview — reuse the upstream memref.subview's result type
    //    verbatim (already rank-reduced when the subview is rank-reducing).
    auto subviewResultType = cast<MemRefType>(subviewOp.getResult().getType());
    auto loomSubviewOp = loom::SubviewOp::create(
        rewriter, loc, subviewResultType, subviewOp.getSource(),
        subviewOp.getOffsets(), subviewOp.getSizes(), subviewOp.getStrides(),
        subviewOp.getStaticOffsets(), subviewOp.getStaticSizes(),
        subviewOp.getStaticStrides(), false, false, false);

    // 2. loom.bufferize_to_memref: pure view of the L1 tensor as a memref.
    Value srcTensor = toMemrefOp.getSource();
    auto srcTensorType = llvm::cast<RankedTensorType>(srcTensor.getType());
    auto l1MemrefType =
        MemRefType::get(srcTensorType.getShape(), srcTensorType.getElementType());
    auto bufToMemref = loom::BufferizeToMemrefOp::create(
        rewriter, loc, l1MemrefType, srcTensor);

    // 3. loom.copy: physically move data L1→DRAM.
    auto l1Symbol = SymbolRefAttr::get(rewriter.getContext(), "mem_L1");
    auto dramSymbol = SymbolRefAttr::get(rewriter.getContext(), "mem_DRAM");
    auto loomCopyOp = loom::CopyOp::create(
        rewriter, loc, bufToMemref.getResult(), loomSubviewOp.getResult(),
        l1Symbol, dramSymbol,
        rewriter.getI64ArrayAttr({1, 1}),
        Value{}, Value{}, Value{}, Value{});

    // 4. Move semaphore_give to after the copy if it exists.
    auto vbIt = tensorToVBIdMap.find(srcTensor);
    if (vbIt != tensorToVBIdMap.end()) {
      Value semaphore = vbIdToSemaphoreMap.lookup(vbIt->second);
      if (semaphore) {
        for (Operation *user : semaphore.getUsers()) {
          if (isa<loom::SemaphoreGiveOp>(user)) {
            user->moveAfter(loomCopyOp);
          }
        }
      }
    }

    // Erase original ops
    rewriter.eraseOp(op);
    if (toMemrefOp->use_empty())
      rewriter.eraseOp(toMemrefOp);
    if (subviewOp->use_empty())
      rewriter.eraseOp(subviewOp);

    return success();
  }

private:
  const llvm::DenseMap<Value, int> &tensorToVBIdMap;
  const llvm::DenseMap<int, Value> &vbIdToSemaphoreMap;
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
    manageSemaphoreTakes();
    anchorVirtualBuffers();
    manageSemaphoreGives();
    applyPatternRewrites();
  }

private:
  /// Trace the outs operand of a linalg op back to its defining loom.alloc.
  /// After tensor canonicalization, handoff anchors are loom.bufferize_to_tensor
  /// ops. Keep a compatibility fallback for older standalone inputs below.
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
      if (auto initOp = current.getDefiningOp<loom::InitTensorOp>()) {
        current = initOp.getBuffer();
        continue;
      }
      if (auto copyOp = current.getDefiningOp<loom::CopyToTensorOp>()) {
        current = copyOp.getBuffer();
        continue;
      }
      // See through semaphore to find the underlying alloc
      if (auto semOp = current.getDefiningOp<loom::SemaphoreTakeOp>()) {
        current = semOp.getSource();
        continue;
      }
      if (auto toTensorOp = current.getDefiningOp<loom::BufferizeToTensorOp>()) {
        current = toTensorOp.getSource();
        continue;
      }
      if (auto toTensorOp = current.getDefiningOp<bufferization::ToTensorOp>()) {
        current = toTensorOp.getBuffer();
        continue;
      }

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

      if (auto allocOp = current.getDefiningOp<loom::AllocOp>())
        return current;

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
        if (auto forOp = dyn_cast<scf::ForOp>(
                blockArg.getOwner()->getParentOp())) {
          unsigned argIdx = blockArg.getArgNumber() - 1; // 1 IV in scf.for
          current = forOp.getInits()[argIdx];
          continue;
        }
      }
      break;
    }
    return nullptr;
  }

  Value getOrCreateInitTensor(Value allocVal, Value semaphore) {
    if (auto it = semaphoreToInitTensor.find(semaphore);
        it != semaphoreToInitTensor.end())
      return it->second;

    OpBuilder b(context);
    if (!semaphore || !semaphore.getDefiningOp())
      return nullptr;
    b.setInsertionPointAfter(semaphore.getDefiningOp());

    auto memrefType = cast<MemRefType>(allocVal.getType());
    auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                            memrefType.getElementType());

    SmallVector<Value> dynamicSizes;
    if (auto allocOp = allocVal.getDefiningOp<loom::AllocOp>())
      dynamicSizes = allocOp.getSizes();

    auto initOp = loom::InitTensorOp::create(
        b, allocVal.getLoc(), tensorType, semaphore, dynamicSizes,
        b.getDenseI64ArrayAttr(memrefType.getShape()));
    semaphoreToInitTensor[semaphore] = initOp.getResult();
    return initOp.getResult();
  }

  void manageSemaphoreTakes() {
    for (const auto &[sig, bucket] : analysisCtx.getBuckets()) {
      for (const auto &vb : bucket.virtualBuffers) {
        if (vb->members.empty())
          continue;

        Value allocVal = colorToAlloc.lookup({sig, vb->color});
        if (!allocVal)
          continue;

        // 1. Determine Take Ops
        int birthIdx = vb->liveness.birth;
        Operation *birthOp = analysisCtx.getOpFromIndex(birthIdx);

        if (!birthOp)
          continue;

        OpBuilder builder(context);
        Location loc = birthOp->getLoc();

        // Step 1: Take Insertion
        builder.setInsertionPointAfter(allocVal.getDefiningOp());
        auto memrefType = cast<MemRefType>(allocVal.getType());
        Value semaphore =
            loom::SemaphoreTakeOp::create(builder, loc, memrefType, allocVal);
        vbIdToSemaphoreMap[vb->id] = semaphore;
      }
    }
  }

  void manageSemaphoreGives() {
    for (const auto &[sig, bucket] : analysisCtx.getBuckets()) {
      for (const auto &vb : bucket.virtualBuffers) {
        if (vb->members.empty())
          continue;

        Value semaphore = vbIdToSemaphoreMap.lookup(vb->id);
        if (!semaphore)
          continue;

        // 2. Determine Give Ops
        int deathIdx = vb->liveness.death;
        Operation *deathOp = analysisCtx.getOpFromIndex(deathIdx);

        if (!deathOp)
          continue;

        Location loc = deathOp->getLoc();

        // Step 2: Give Insertion
        Operation *insertionPoint = deathOp;
        bool insertAfter = true;

        // Scope hoisting: For Eternal and Fused VBs, their lifetime spans
        // across inner loop iterations. If their last use (deathOp) is inside a
        // nested loop, we must hoist their death to after that loop, instead of
        // killing them inside.
        if (vb->type == VBType::Eternal || vb->type == VBType::Fused) {
          while (insertionPoint->getParentOp() != bucket.scopeOp) {
            insertionPoint = insertionPoint->getParentOp();
            insertAfter = true;
          }
        }

        // Terminator guard: never insert AFTER a terminator
        if (insertionPoint->hasTrait<OpTrait::IsTerminator>()) {
          insertAfter = false;
        }

        OpBuilder deathBuilder(context);
        if (insertAfter) {
          deathBuilder.setInsertionPointAfter(insertionPoint);
        } else {
          deathBuilder.setInsertionPoint(insertionPoint);
        }

        loom::SemaphoreGiveOp::create(deathBuilder, loc, semaphore);
      }
    }
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

      int numColors = plan.colorCountPerBucket.lookup(sig);
      for (int c = 0; c < numColors; ++c) {
        Operation *earliestBirthOp = nullptr;
        int minBirthIdx = std::numeric_limits<int>::max();

        for (const auto &vb : bucket.virtualBuffers) {
          if (vb->members.empty() || vb->color != c)
            continue;
          if (vb->liveness.birth < minBirthIdx) {
            minBirthIdx = vb->liveness.birth;
          }
        }

        if (minBirthIdx != std::numeric_limits<int>::max()) {
          earliestBirthOp = analysisCtx.getOpFromIndex(minBirthIdx);
        }

        // Compute dynamic/static sizes first so we can check dominance below.
        SmallVector<Value> dynamicSizes;
        SmallVector<int64_t> staticSizes;
        for (const auto &dim : sig.dims) {
          if (auto attr = dim.dyn_cast<Attribute>()) {
            int64_t v = cast<IntegerAttr>(attr).getInt();
            if (v < 0) {
              bucket.scopeOp->emitError()
                  << "allocation planning found unresolved dynamic dimension "
                  << "in shape signature";
              assert(false &&
                     "allocation planning found unresolved dynamic dimension");
              llvm::report_fatal_error(
                  "allocation planning found unresolved dynamic dimension");
            }
            staticSizes.push_back(v);
          } else {
            staticSizes.push_back(ShapedType::kDynamic);
            dynamicSizes.push_back(dim.dyn_cast<Value>());
          }
        }

        OpBuilder builder(context);
        builder.setInsertionPointToStart(&bucket.scopeOp->getRegion(0).front());

        if (earliestBirthOp) {
          // Trace up to immediate child of scopeOp to ensure we stay in the same block
          Operation *insertPt = earliestBirthOp;
          while (insertPt && insertPt->getParentOp() != bucket.scopeOp) {
            insertPt = insertPt->getParentOp();
          }
          if (insertPt) {
            // If any dynamic size is defined *inside* insertPt, placing the
            // alloc before insertPt would violate SSA dominance. Instead,
            // insert just before the birth op itself — all dynamic sizes are
            // guaranteed available there (e.g. gather inside scf.if).
            bool sizeInsideInsertPt = llvm::any_of(dynamicSizes, [&](Value dynSize) {
              return dynSize.getDefiningOp() &&
                     insertPt->isProperAncestor(dynSize.getDefiningOp());
            });
            if (sizeInsideInsertPt) {
              builder.setInsertionPoint(earliestBirthOp);
            } else {
              builder.setInsertionPoint(insertPt);
            }
          }
        }
        // For 0-rank tensors (scalars), the result memref stays 0-rank so that
        // downstream ops (semaphore_take, copy, bufferize_to_tensor) are
        // unchanged. The static_sizes attribute records [1] so that
        // formatAllocDims produces a correct 1-element footprint in the ETG.
        SmallVector<int64_t> allocAttrSizes = staticSizes;
        if (allocAttrSizes.empty() && dynamicSizes.empty())
          allocAttrSizes.push_back(1);
        auto allocType = MemRefType::get(staticSizes, sig.elementType);
        auto allocOp = loom::AllocOp::create(
            builder, bucket.scopeOp->getLoc(), allocType, dynamicSizes,
            builder.getDenseI64ArrayAttr(allocAttrSizes), nullptr,
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
        Operation *loopOp = nullptr;
        if (vb->type == VBType::Fused || vb->type == VBType::LoopCarried) {
          for (TensorNode *member : vb->members) {
            if (auto blockArg = mlir::dyn_cast<BlockArgument>(member->value)) {
              iterArg = blockArg;
              loopOp = blockArg.getOwner()->getParentOp();
              break;
            }
          }
        }

        Value semaphore = vbIdToSemaphoreMap.lookup(vb->id);
        if (!semaphore)
          continue;

        for (TensorNode *member : vb->members) {
          // Handle any DPS op (linalg AND loom ops like loom.gather).
          auto dpsOp =
              dyn_cast_or_null<mlir::DestinationStyleOpInterface>(member->definingOp);
          if (dpsOp) {
            for (OpOperand &outsOp : dpsOp.getDpsInitsMutable()) {
              if (outsOp.get().getDefiningOp<tensor::EmptyOp>()) {
                // If this VB has an iterarg AND the op is inside the loop,
                // use iterarg as the outs to preserve SSA chain.
                if (iterArg && loopOp &&
                    loopOp->isProperAncestor(member->definingOp)) {
                  outsOp.set(iterArg);
                } else {
                  outsOp.set(getOrCreateInitTensor(allocVal, semaphore));
                }
              }
            }
            continue;
          }

          // loom.broadcast is DPS-like but does not implement DSOI in our setup.
          if (auto broadcastOp =
                  dyn_cast_or_null<loom::BroadcastOp>(member->definingOp)) {
            Value init = broadcastOp.getInit();
            if (!init.getDefiningOp<tensor::EmptyOp>())
              continue;

            Value newInit;
            if (iterArg && loopOp && loopOp->isProperAncestor(broadcastOp))
              newInit = iterArg;
            else
              newInit = getOrCreateInitTensor(allocVal, semaphore);
            broadcastOp.getInitMutable().assign(newInit);
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
          // IterArg: trace to init value, clone its defining op onto PB.
          // Helper lambda to redirect loop init after cloning.
          auto redirectLoopInit = [&](unsigned argIdx, ValueRange inits,
                                      auto loopOp) {
            Value initVal = inits[argIdx];
            Operation *initDefOp = initVal.getDefiningOp();
            if (initDefOp && isa<linalg::LinalgOp>(initDefOp)) {
              Value semaphore = vbIdToSemaphoreMap.lookup(vb->id);
              if (!semaphore)
                return;
              Value initTensor = getOrCreateInitTensor(allocVal, semaphore);
              OpBuilder cloneBuilder(context);
              cloneBuilder.setInsertionPointAfter(initDefOp);
              Operation *clonedOp = cloneBuilder.clone(*initDefOp);
              cast<linalg::LinalgOp>(clonedOp).getDpsInitsMutable()[0].set(
                  initTensor);
              clonedOp->getResult(0).setType(initTensor.getType());
              loopOp.getInitsMutable()[argIdx].set(clonedOp->getResult(0));
            }
          };

          Operation *parentOp = blockArg.getOwner()->getParentOp();
          if (auto affFor = dyn_cast<affine::AffineForOp>(parentOp)) {
            unsigned argIdx = blockArg.getArgNumber() -
                              affFor.getBody()->getNumArguments() +
                              affFor.getNumIterOperands();
            redirectLoopInit(argIdx, affFor.getInits(), affFor);
          } else if (auto scfFor = dyn_cast<scf::ForOp>(parentOp)) {
            unsigned argIdx = blockArg.getArgNumber() - 1;
            redirectLoopInit(argIdx, scfFor.getInits(), scfFor);
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
              Value semaphore = vbIdToSemaphoreMap.lookup(vb->id);
              if (semaphore) {
                Value newOuts = getOrCreateInitTensor(allocVal, semaphore);
                auto copyOp = linalg::CopyOp::create(
                    builder, linalgOp.getLoc(), outsTensor, newOuts);
                outsOperand.set(copyOp.getResult(0));
                // Update result type to match new init_tensor
                linalgOp->getResult(0).setType(newOuts.getType());
              }
            } else if (!currentAlloc) {
              // No known source buffer (e.g. tensor.empty or external subview)
              Value semaphore = vbIdToSemaphoreMap.lookup(vb->id);
              if (semaphore) {
                outsOperand.set(getOrCreateInitTensor(allocVal, semaphore));
                linalgOp->getResult(0).setType(outsOperand.get().getType());
              }
            }
          } else if (auto broadcastOp = dyn_cast_or_null<loom::BroadcastOp>(defOp)) {
            Value semaphore = vbIdToSemaphoreMap.lookup(vb->id);
            if (!semaphore)
              continue;

            Value outsTensor = broadcastOp.getInit();
            Value currentAlloc = traceOutsToAlloc(outsTensor);
            if (currentAlloc && currentAlloc != allocVal) {
              broadcastOp.getInitMutable().assign(
                  getOrCreateInitTensor(allocVal, semaphore));
            } else if (!currentAlloc) {
              broadcastOp.getInitMutable().assign(
                  getOrCreateInitTensor(allocVal, semaphore));
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

        // Find the yield op (supports both affine.for and scf.for)
        Operation *parentOp =
            mlir::cast<BlockArgument>(split.iterArg).getOwner()->getParentOp();
        Operation *yieldOp = nullptr;
        if (auto affFor = dyn_cast<affine::AffineForOp>(parentOp))
          yieldOp = affFor.getBody()->getTerminator();
        else if (auto scfFor = dyn_cast<scf::ForOp>(parentOp))
          yieldOp = scfFor.getBody()->getTerminator();
        else
          continue;

        OpBuilder builder(context);
        builder.setInsertionPoint(yieldOp);

        auto copyOp = linalg::CopyOp::create(builder, yieldOp->getLoc(),
                                              yieldVal, itArg->first);

        // Update the yield operand
        yieldOp->setOperand(split.iterArgIndex, copyOp.getResult(0));
      }
    }
  }

  void applyPatternRewrites() {
    // Build a temporary tensor-to-VB mapping for the pattern
    llvm::DenseMap<Value, int> tensorToVBIdMap;
    for (const auto &[sig, bucket] : analysisCtx.getBuckets()) {
      for (const auto &vb : bucket.virtualBuffers) {
        for (TensorNode *member : vb->members) {
          tensorToVBIdMap[member->value] = vb->id;
        }
      }
    }

    RewritePatternSet patterns(context);
    patterns.add<ReadBlockLoadingLowering>(context, colorToAlloc,
                                           plan.tensorToBufferMap,
                                           tensorToVBIdMap, vbIdToSemaphoreMap);
    patterns.add<WriteBackLowering>(context, tensorToVBIdMap,
                                    vbIdToSemaphoreMap);

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
  llvm::DenseMap<Value, Value> semaphoreToInitTensor;
  llvm::DenseMap<int, Value> vbIdToSemaphoreMap;
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
