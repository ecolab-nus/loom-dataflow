/**
 * @file TileLoomToTTKernel.cpp
 * @brief Main pass for converting TileLoom IR to TTKernel dialect.
 */

#include "ComputeOpToTTKernel.h"
#include "MemoryOpToTTKernel.h"
#include "SCFOpToTTKernel.h"
#include "FuncOpToTTKernel.h"
#include "TileLoomToTTKernel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
// TTKernel dialect (tt-mlir) for types/ops created by conversion patterns.
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/SetVector.h"

// Loom dialect headers for ::loom::AllocOp, ::loom::CopyOp, etc.
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;
using namespace mlir::loom;
using namespace tt::ttkernel;

namespace {

/**
 * @brief Type converter for TileLoom to TTKernel conversion.
 * 
 * @details This type converter handles the conversion of memref types
 *          to TTKernel CB types. MemRef types are converted to CBType,
 *          which is the circular buffer type used in TTKernel dialect.
 */
class TileLoomTypeConverter : public TypeConverter {
public:
  TileLoomTypeConverter() {
    // Default conversion: keep types as-is if no specific conversion matches
    addConversion([](Type type) { return type; });
    
    // Convert MemRefType to CBType for circular buffers
    addConversion([](MemRefType memref) -> Type {
      // Convert memref to CBType. The CBType wraps the memref and stores
      // the number of elements and element type.
      return CBType::get(memref);
    });
    
  }
};

/**
 * @brief Erase all Dataflow dialect operations from the module.
 *
 * @details This helper walks the entire module and removes every operation
 *          whose dialect namespace is "df". These operations are part of the
 *          hardware description in the Dataflow dialect and are not needed
 *          after TileLoom has been fully lowered to the TTKernel dialect.
 *
 * @param module The module in which to erase all Dataflow dialect operations.
 */
static void eraseAllDfOps(ModuleOp module) {
  SmallVector<Operation *, 16> toErase;
  module.walk([&](Operation *op) {
    Dialect *dialect = op->getDialect();
    if (dialect && dialect->getNamespace() == StringRef("df"))
      toErase.push_back(op);
  });

  // Erase in reverse to avoid invalidating the IR while deleting.
  for (auto *op : llvm::reverse(toErase))
    op->erase();
}

static bool hasResultUsersOutside(Operation *op, Operation *scopeOp) {
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (!scopeOp->isAncestor(user))
        return true;
    }
  }
  return false;
}

static bool isHostArtifactLoop(scf::ForOp forOp) {
  for (Operation &inner : forOp.getBody()->without_terminator()) {
    if (!isa<mlir::tt::ttkernel::GetArgValOp, arith::IndexCastOp>(inner))
      return false;
    if (hasResultUsersOutside(&inner, forOp))
      return false;
  }
  return true;
}

static void eraseHostLoweringArtifacts(ModuleOp module) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    StringRef name = func.getName();
    if (!name.ends_with("__host") && !name.ends_with("__writer") &&
        !name.ends_with("__reader"))
      continue;

    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SetVector<Operation *> opsToErase;

      func.walk([&](scf::ForOp forOp) {
        if (isHostArtifactLoop(forOp))
          opsToErase.insert(forOp);
      });

      func.walk([&](mlir::tt::ttkernel::GetArgValOp getArgOp) {
        if (getArgOp.getResult().use_empty())
          opsToErase.insert(getArgOp.getOperation());
      });

      func.walk([&](arith::IndexCastOp castOp) {
        if (castOp.getResult().use_empty())
          opsToErase.insert(castOp);
      });

      if (opsToErase.empty())
        break;

      changed = true;
      for (Operation *op : llvm::reverse(opsToErase))
        op->erase();
    }
  }
}

static LogicalResult rewriteBatch1MatmulToMatmul(ModuleOp module) {
  SmallVector<linalg::BatchMatmulOp> batchMatmuls;
  module.walk([&](linalg::BatchMatmulOp op) { batchMatmuls.push_back(op); });

  for (linalg::BatchMatmulOp op : batchMatmuls) {
    auto lhsTy = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto rhsTy = dyn_cast<MemRefType>(op.getInputs()[1].getType());
    auto outTy = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    if (!lhsTy || !rhsTy || !outTy || !lhsTy.hasStaticShape() ||
        !rhsTy.hasStaticShape() || !outTy.hasStaticShape() ||
        lhsTy.getRank() != 3 || rhsTy.getRank() != 3 ||
        outTy.getRank() != 3) {
      return op.emitOpError(
          "expected static-rank memref<1xMxK>, memref<1xKxN>, memref<1xMxN>");
    }

    if (lhsTy.getShape()[0] != 1 || rhsTy.getShape()[0] != 1 ||
        outTy.getShape()[0] != 1) {
      return op.emitOpError(
          "only batch-size 1 linalg.batch_matmul is supported");
    }

    SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}};
    auto makeCollapsedType = [&](MemRefType srcTy) {
      return MemRefType::get({srcTy.getShape()[1], srcTy.getShape()[2]},
                             srcTy.getElementType(), AffineMap(),
                             srcTy.getMemorySpace());
    };

    OpBuilder builder(op);
    Value lhs2D = builder.create<memref::CollapseShapeOp>(
        op.getLoc(), makeCollapsedType(lhsTy), op.getInputs()[0],
        reassociation);
    Value rhs2D = builder.create<memref::CollapseShapeOp>(
        op.getLoc(), makeCollapsedType(rhsTy), op.getInputs()[1],
        reassociation);
    Value out2D = builder.create<memref::CollapseShapeOp>(
        op.getLoc(), makeCollapsedType(outTy), op.getOutputs()[0],
        reassociation);

    builder.create<linalg::MatmulOp>(op.getLoc(), ValueRange{lhs2D, rhs2D},
                                     ValueRange{out2D});
    op.erase();
  }

  return success();
}

/**
 * @brief Pass that converts TileLoom IR to TTKernel dialect.
 * 
 * @details This pass applies conversion patterns to transform TileLoom
 *          operations (e.g., loom.alloc, loom.copy) into
 *          TTKernel operations. It converts memref types to CBType and
 *          handles function signature conversion.
 */
class TileLoomToTTKernelPass
    : public PassWrapper<TileLoomToTTKernelPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileLoomToTTKernelPass)

  StringRef getArgument() const override {
    return "loom-tileloom-to-ttkernel";
  }

  StringRef getDescription() const override {
    return "Convert TileLoom IR operations to TTKernel dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect,
                    linalg::LinalgDialect>();
    // Ensure TTKernel is loaded before patterns create TTKernel types (e.g.,
    // DataFormatType). Otherwise, creating such types can fail with
    // "storage uniquer isn't initialized".
    registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
    // Ensure EmitC dialect is loaded for verbatim operations in host functions.
    registry.insert<mlir::emitc::EmitCDialect>();
    // Ensure Loom dialect is loaded for loom.alloc, loom.copy operations.
    registry.insert<::loom::LoomDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Preprocessing: hoist invariant allocs and simplify before lowering.
    //
    // This is useful to move loop-invariant `memref.alloc` ops outward (e.g.,
    // out of `scf.for` nests) so the later TTKernel lowering sees fewer
    // repeated allocations.
    PassManager prePm(context);
    prePm.addNestedPass<func::FuncOp>(
        bufferization::createBufferLoopHoistingPass());
    prePm.addNestedPass<func::FuncOp>(
        bufferization::createBufferHoistingPass());
    //prePm.addPass(createCanonicalizerPass());
    if (failed(prePm.run(module))) {
      signalPassFailure();
      return;
    }

    // Step 2: Specialize functions into compute/reader/writer variants.
    // This clones each function into:
    // - `__compute`: stores erased (compute-only)
    // - `__reader` : stores + compute erased (loads-only)
    // - `__writer` : loads  + compute erased (stores-only)
    //
    // This must run before MemoryOp/ComputeOp lowering so each specialized
    // function is lowered independently.
    specializeFunctionsForTTKernel(module);

    // Create shared compile-arg tracker for index management.
    auto compileArgTracker = std::make_shared<CompileArgTracker>();

    // Create type converter (needed for memref -> CB conversion).
    TileLoomTypeConverter typeConverter;

    // Step 3: Run a per-function DSE-style cleanup after splitting.
    //
    // The specialization step above can leave behind dead ops (e.g. allocs,
    // casts, unused loop-carried values). Run a small cleanup pipeline on each
    // function to simplify IR before the TTKernel lowering patterns run.
    // NOTE: This MUST run BEFORE replaceFuncArgsWithCompileArgs because
    // the cleanup passes may modify the IR, which would invalidate any
    // values stored in the tracker.
    PassManager postSplitPm(context);
    postSplitPm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    postSplitPm.addNestedPass<func::FuncOp>(createCSEPass());
    // Symbol DCE cleans up any now-unreferenced symbols at the module level.
    postSplitPm.addPass(createSymbolDCEPass());
    if (failed(postSplitPm.run(module))) {
      signalPassFailure();
      return;
    }

    if (failed(rewriteBatch1MatmulToMatmul(module))) {
      signalPassFailure();
      return;
    }

    // Step 3.5: Replace index-type function arguments with GetArgValOp.
    // For each function, insert GetArgValOp at the beginning of the
    // function body to materialize compile-time arguments:
    // - Index types: GetArgValOp returning i32, then cast to index
    // - Memref types (DRAM pointers): create CB and base address values
    // This is done AFTER cleanup passes to avoid dangling value references.
    OpBuilder builder(context);
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(replaceFuncArgsWithCompileArgs(func, compileArgTracker,
                                                 typeConverter, builder))) {
        signalPassFailure();
        return;
      }
    }
    // Note: We don't remove arguments here. Memref args are still used by
    // reinterpret_cast ops. They will be removed after conversion when all
    // uses are eliminated.


    // Set up conversion target
    ConversionTarget target(*context);
    
    // Mark loom.alloc, loom.semaphore, and loom.copy as illegal
    // (needs conversion).
    //target.addIllegalOp<::loom::AllocOp>();
    target.addIllegalOp<::loom::SemaphoreOp>();
    target.addIllegalOp<::loom::CopyOp>();
    
    // Mark memref operations that don't need conversion as legal
    // (they will be type-converted automatically)
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect>();

    // SCF dialect is generally legal, but we require a conversion for
    // scf.parallel so it can be lowered to straight-line code with
    // compile-time iterators.
    target.addDynamicallyLegalOp<scf::ParallelOp>(
        [&](scf::ParallelOp op) {
          // Always mark scf.parallel as illegal so our conversion runs.
          // If a loop is not rewritten, the conversion will fail.
          (void)op;
          return false;
        });

    // linalg dialect is generally legal, but we require a conversion for
    // linalg.matmul so it can be lowered to TTKernel matmul.
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addDynamicallyLegalOp<linalg::MatmulOp>(
        [&](linalg::MatmulOp op) { return false; });
    target.addDynamicallyLegalOp<linalg::BatchMatmulOp>(
        [&](linalg::BatchMatmulOp op) { return false; });
    target.addDynamicallyLegalOp<linalg::FillOp>(
        [&](linalg::FillOp op) { return false; });
    target.addDynamicallyLegalOp<linalg::GenericOp>(
        [&](linalg::GenericOp op) {
          return !mlir::loom::isSupportedFlashAttentionGeneric(op);
        });
    target.addDynamicallyLegalOp<linalg::CopyOp>(
        [&](linalg::CopyOp op) {
          return !mlir::loom::shouldConvertComputeLinalgCopy(op);
        });
    target.addDynamicallyLegalOp<::loom::CopyOp>(
        [&](::loom::CopyOp op) {
          return false;
        });
    
    // Mark module and function ops as legal (they will be type-converted)
    target.addLegalOp<ModuleOp>();
    
    // Function ops need special handling for signature conversion
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      // Check if function signature needs conversion
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    
    // Mark memref.reinterpret_cast as illegal (consumed by conversion patterns)
    target.addDynamicallyLegalOp<memref::ReinterpretCastOp>(
        [&](memref::ReinterpretCastOp op) {
          return false;
        });
    target.addDynamicallyLegalOp<memref::CollapseShapeOp>(
        [&](memref::CollapseShapeOp op) {
          auto srcTy = dyn_cast<MemRefType>(op.getSrcType());
          auto dstTy = dyn_cast<MemRefType>(op.getResultType());
          if (!srcTy || !dstTy || !srcTy.hasStaticShape() ||
              !dstTy.hasStaticShape())
            return true;
          return srcTy.getNumElements() != dstTy.getNumElements();
        });
    target.addLegalDialect<mlir::tt::ttkernel::TTKernelDialect>();

    // Populate conversion patterns
    RewritePatternSet patterns(context);

    // Ensure function signatures, calls, and returns are rewritten when types
    // (e.g., index -> i32) change.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    
    // Add SCF operation conversion patterns (e.g., scf.parallel -> CT args)
    populateSCFOpConversionPatterns(patterns, typeConverter, context,
                                    compileArgTracker);

    // Add memory operation conversion patterns (loom.alloc / loom.copy)
    populateMemoryOpConversionPatterns(patterns, typeConverter, context,
                                       compileArgTracker);
    // Add compute operation conversion patterns (e.g., linalg.matmul)
    populateComputeOpConversionPatterns(patterns, typeConverter, context);

    // Apply conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Host specialization can leave loop shells containing only unused
    // get_arg_val/index_cast artifacts. Remove them before final cleanup.
    eraseHostLoweringArtifacts(module);

    // Post-conversion: Remove all function arguments.
    // After conversion, memref args used in reinterpret_cast should be dead
    // (the conversion patterns emit GetArgValOp for base addresses).
    // Index args were already replaced before conversion.
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(removeAllFunctionArguments(func))) {
        // Some argument still has uses - emit a warning but continue.
        // This might happen if some conversion pattern didn't run.
        func.emitWarning() << "could not remove all function arguments";
      }
    }


    // postprocessing optimizations
    PassManager postPm(context);
    postPm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    postPm.addNestedPass<func::FuncOp>(createCSEPass());
    if (failed(postPm.run(module))) {
      signalPassFailure();
      return;
    }

    // Final stage: strip all Dataflow (df) dialect ops from the module.
    eraseAllDfOps(module);
  }
};

/**
 * @brief Post-EmitC host signature rewrite pass.
 *
 * @details Rewrites each `__host` function signature to:
 *          - one `std::vector<bfloat16>&` per original memref argument
 *          - one trailing `IDevice*` as the final argument
 *
 * The number of memref-backed host arguments is read from the
 * `loom.host_memref_count` attribute emitted during host construction.
 */
class PostEmitCHostSignaturePass
    : public PassWrapper<PostEmitCHostSignaturePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PostEmitCHostSignaturePass)

  StringRef getArgument() const override {
    return "loom-post-emitc-host-signature";
  }

  StringRef getDescription() const override {
    return "Rewrite __host signature to vector inputs, core range args, plus trailing IDevice*";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, emitc::EmitCDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    auto hostVectorType =
        emitc::OpaqueType::get(ctx, "std::vector<bfloat16>&");
    auto coreCoordType = emitc::OpaqueType::get(ctx, "uint32_t");
    auto deviceType = emitc::PointerType::get(
        emitc::OpaqueType::get(ctx, "IDevice"));

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!func.getName().ends_with("__host"))
        continue;

      auto countAttr =
          func->getAttrOfType<IntegerAttr>("loom.host_memref_count");
      if (!countAttr) {
        func.emitError()
            << "missing required attribute 'loom.host_memref_count'";
        signalPassFailure();
        return;
      }

      int64_t hostMemrefCount = countAttr.getInt();
      if (hostMemrefCount < 0) {
        func.emitError() << "invalid negative 'loom.host_memref_count': "
                         << hostMemrefCount;
        signalPassFailure();
        return;
      }

      Block &entry = func.front();
      for (BlockArgument arg : entry.getArguments()) {
        if (!arg.use_empty()) {
          func.emitError()
              << "cannot rewrite host signature; argument "
              << arg.getArgNumber() << " still has uses";
          signalPassFailure();
          return;
        }
      }

      for (int64_t idx = static_cast<int64_t>(entry.getNumArguments()) - 1;
           idx >= 0; --idx) {
        entry.eraseArgument(static_cast<unsigned>(idx));
      }

      SmallVector<Type> newInputs;
      for (int64_t i = 0; i < hostMemrefCount; ++i)
        newInputs.push_back(hostVectorType);
      newInputs.push_back(coreCoordType); // start_core_x
      newInputs.push_back(coreCoordType); // start_core_y
      newInputs.push_back(coreCoordType); // end_core_x
      newInputs.push_back(coreCoordType); // end_core_y
      newInputs.push_back(deviceType);

      for (Type inputType : newInputs)
        entry.addArgument(inputType, func.getLoc());

      auto funcType = func.getFunctionType();
      func.setType(
          FunctionType::get(ctx, newInputs, funcType.getResults()));
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::loom::createTileLoomToTTKernelPass() {
  return std::make_unique<TileLoomToTTKernelPass>();
}

void mlir::loom::registerTileLoomToTTKernelPass() {
  PassRegistration<TileLoomToTTKernelPass>();
  PassRegistration<PostEmitCHostSignaturePass>();
}
