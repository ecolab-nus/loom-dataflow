/**
 * @file TileLoomToTTKernel.cpp
 * @brief Main pass for converting TileLoom IR to TTKernel dialect.
 */

#include "ComputeOpToTTKernel.h"
#include "MemoryOpToTTKernel.h"
#include "AffineOpToTTKernel.h"
#include "FuncOpToTTKernel.h"
#include "TileLoomToTTKernel.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
// TTKernel dialect (tt-mlir) for types/ops created by conversion patterns.
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

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
    
/*     // Add target materialization for CBType -> MemRefType (if needed)
    addTargetMaterialization([](OpBuilder &builder, Type type,
                                 ValueRange inputs, Location loc) -> Value {
      // If we need to materialize a memref from a CBType, we can extract
      // the underlying memref. For now, return nullptr to indicate no
      // materialization is available.
      return nullptr;
    });
    
    // Add source materialization for MemRefType -> CBType (if needed)
    addSourceMaterialization([](OpBuilder &builder, Type type,
                                 ValueRange inputs, Location loc) -> Value {
      // If we have a memref value that needs to be converted to CBType,
      // we can wrap it. For now, return nullptr to indicate no
      // materialization is available.
      return nullptr;
    }); */
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

/**
 * @brief Pass that converts TileLoom IR to TTKernel dialect.
 * 
 * @details This pass applies conversion patterns to transform TileLoom
 *          operations (e.g., memref.copy with loom.copy.choice) into
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
                    affine::AffineDialect, linalg::LinalgDialect>();
    // Ensure TTKernel is loaded before patterns create TTKernel types (e.g.,
    // DataFormatType). Otherwise, creating such types can fail with
    // "storage uniquer isn't initialized".
    registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Preprocessing: hoist invariant allocs and simplify before lowering.
    //
    // This is useful to move loop-invariant `memref.alloc` ops outward (e.g.,
    // out of `affine.for` nests) so the later TTKernel lowering sees fewer
    // repeated allocations.
    PassManager prePm(context);
    prePm.addNestedPass<func::FuncOp>(
        bufferization::createBufferLoopHoistingPass());
    prePm.addNestedPass<func::FuncOp>(
        bufferization::createBufferHoistingPass());
    prePm.addNestedPass<func::FuncOp>(
        affine::createAffineLoopInvariantCodeMotionPass());
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

    // Step 3.5: Replace index-type function arguments with GetCompileArgValOp.
    // For each function, insert GetCompileArgValOp at the beginning of the
    // function body to materialize compile-time arguments:
    // - Index types: GetCompileArgValOp returning i32, then cast to index
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
    
    // Classify memref.copy as load/store based on whether the target is a
    // reinterpret_cast. Always mark them illegal so the lowering patterns run.
    target.addDynamicallyLegalOp<memref::CopyOp>([&](memref::CopyOp op) {
      bool sourceIsRC =
          op.getSource().getDefiningOp<memref::ReinterpretCastOp>() != nullptr;
      bool targetIsRC =
          op.getTarget().getDefiningOp<memref::ReinterpretCastOp>() != nullptr;
      if (sourceIsRC || targetIsRC) {
        return false;
      } else {
        return true;
      }
    });
    
    // Mark all memref.alloc as illegal (needs conversion)
    // The type converter converts memref to CB, so all allocs need conversion.
    target.addIllegalOp<memref::AllocOp>();
    
    // Mark memref operations that don't need conversion as legal
    // (they will be type-converted automatically)
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect>();

    // Affine dialect is generally legal, but we require a conversion for
    // affine.parallel so it can be lowered to straight-line code with
    // compile-time iterators.
    target.addLegalDialect<affine::AffineDialect>();
    target.addDynamicallyLegalOp<affine::AffineParallelOp>(
        [&](affine::AffineParallelOp op) {
          // Always mark affine.parallel as illegal so our conversion runs.
          // If a loop is not rewritten, the conversion will fail.
          (void)op;
          return false;
        });

    // linalg dialect is generally legal, but we require a conversion for
    // linalg.matmul so it can be lowered to TTKernel matmul.
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addDynamicallyLegalOp<linalg::MatmulOp>(
        [&](linalg::MatmulOp op) { return false; });
    
    // Mark module and function ops as legal (they will be type-converted)
    target.addLegalOp<ModuleOp>();
    
    // Function ops need special handling for signature conversion
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      // Check if function signature needs conversion
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    
    // Mark memref operations that don't have loom.copy.choice as legal
    target.addDynamicallyLegalOp<memref::ReinterpretCastOp>(
        [&](memref::ReinterpretCastOp op) {
          return false;
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

    
    // Add Affine operation conversion patterns (e.g., affine.parallel -> CT args)
    populateAffineOpConversionPatterns(patterns, typeConverter, context,
                                       compileArgTracker);

    // Add memory operation conversion patterns (memref.copy with loom.copy.choice)
    populateMemoryOpConversionPatterns(patterns, typeConverter, context,
                                       compileArgTracker);
    // Add compute operation conversion patterns (e.g., linalg.matmul)
    populateComputeOpConversionPatterns(patterns, typeConverter, context);

    // Apply conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Post-conversion: Remove all function arguments.
    // After conversion, memref args used in reinterpret_cast should be dead
    // (the conversion patterns emit GetCompileArgValOp for base addresses).
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

} // namespace

std::unique_ptr<Pass> mlir::loom::createTileLoomToTTKernelPass() {
  return std::make_unique<TileLoomToTTKernelPass>();
}

void mlir::loom::registerTileLoomToTTKernelPass() {
  PassRegistration<TileLoomToTTKernelPass>();
}