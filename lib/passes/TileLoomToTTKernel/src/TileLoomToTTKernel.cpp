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
#include "llvm/Support/CommandLine.h"
#include <string>

// Loom dialect headers for ::loom::AllocOp, ::loom::CopyOp, etc.
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomEnums.h.inc"
#include "LoomAttributes.h.inc"
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
 * @brief Erase architecture-description dialect operations from the module.
 *
 * @details This helper walks the module and removes every operation whose
 *          dialect namespace is "df" or "adl". These ops describe topology /
 *          memory hierarchy metadata and are not required after TTKernel
 *          lowering.
 *
 * @param module The module in which to erase descriptor dialect operations.
 */
static void eraseDescriptorDialectOps(ModuleOp module) {
  SmallVector<Operation *, 16> toErase;
  module.walk([&](Operation *op) {
    Dialect *dialect = op->getDialect();
    if (dialect && (dialect->getNamespace() == StringRef("df") ||
                    dialect->getNamespace() == StringRef("adl")))
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

static bool isGeneratedHostFuncName(StringRef name) {
  return name.ends_with("__host") || name.ends_with("__host_cpp") ||
         name.ends_with("__host_pybind");
}

static void eraseHostLoweringArtifacts(ModuleOp module) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    StringRef name = func.getName();
    if (!isGeneratedHostFuncName(name) && !name.ends_with("__writer") &&
        !name.ends_with("__reader") && !name.ends_with("__compute"))
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

static bool isKernelRuntimeSetupCleanupTarget(func::FuncOp func) {
  StringRef name = func.getName();
  return name.ends_with("__reader") || name.ends_with("__compute") ||
         name.ends_with("__writer");
}

static bool isDeadRemovableRuntimeSetupOp(Operation *op) {
  if (!op || op->getNumRegions() != 0)
    return false;

  if (op->getNumResults() == 0)
    return false;

  if (!llvm::all_of(op->getResults(),
                    [](Value result) { return result.use_empty(); }))
    return false;

  return isa<GetSemaphoreOp, TensorAccessorArgsOp, TensorAccessorOp, GetNocAddrOp,
             ExperimentalGetNocMulticastAddrOp, CastToL1PtrOp, GetTileSizeOp,
             GetArgValOp, arith::IndexCastOp>(op);
}

static void eraseDeadRuntimeSetupArtifacts(ModuleOp module) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!isKernelRuntimeSetupCleanupTarget(func))
      continue;

    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SetVector<Operation *> opsToErase;

      func.walk([&](Operation *op) {
        if (isDeadRemovableRuntimeSetupOp(op))
          opsToErase.insert(op);
      });

      if (opsToErase.empty())
        break;

      changed = true;
      for (Operation *op : llvm::reverse(opsToErase))
        op->erase();
    }
  }
}

static void eraseLoomLoopAttrs(ModuleOp module) {
  auto erasePrefixedAttrs = [](Operation *op) {
    SmallVector<StringAttr, 8> attrsToErase;
    for (NamedAttribute namedAttr : op->getAttrs()) {
      if (namedAttr.getName().strref().starts_with("loom."))
        attrsToErase.push_back(namedAttr.getName());
    }
    for (StringAttr attrName : attrsToErase)
      op->removeAttr(attrName);
  };

  module.walk([&](scf::ForOp forOp) { erasePrefixedAttrs(forOp.getOperation()); });
  module.walk([&](scf::ParallelOp parOp) {
    erasePrefixedAttrs(parOp.getOperation());
  });
}

static LogicalResult rewriteMatmulToBatch1Matmul(ModuleOp module) {
  SmallVector<linalg::MatmulOp> matmuls;
  module.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });

  for (linalg::MatmulOp op : matmuls) {
    auto lhsTy = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto rhsTy = dyn_cast<MemRefType>(op.getInputs()[1].getType());
    auto outTy = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    if (!lhsTy || !rhsTy || !outTy || lhsTy.getRank() != 2 ||
        rhsTy.getRank() != 2 || outTy.getRank() != 2)
      continue;

    auto makeBatch1Type = [&](MemRefType srcTy) -> MemRefType {
      int64_t dim0 = srcTy.isDynamicDim(0) ? ShapedType::kDynamic
                                           : srcTy.getDimSize(0);
      int64_t dim1 = srcTy.isDynamicDim(1) ? ShapedType::kDynamic
                                           : srcTy.getDimSize(1);
      return MemRefType::get({1, dim0, dim1}, srcTy.getElementType(),
                             AffineMap(), srcTy.getMemorySpace());
    };

    SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}};
    OpBuilder builder(op);
    Value lhs3D = builder.create<memref::ExpandShapeOp>(
        op.getLoc(), makeBatch1Type(lhsTy), op.getInputs()[0], reassociation);
    Value rhs3D = builder.create<memref::ExpandShapeOp>(
        op.getLoc(), makeBatch1Type(rhsTy), op.getInputs()[1], reassociation);
    Value out3D = builder.create<memref::ExpandShapeOp>(
        op.getLoc(), makeBatch1Type(outTy), op.getOutputs()[0], reassociation);

    builder.create<linalg::BatchMatmulOp>(op.getLoc(),
                                          ValueRange{lhs3D, rhs3D},
                                          ValueRange{out3D});
    op.erase();
  }

  return success();
}

static LogicalResult rewriteLoomLinearAlgebraToLinalg(ModuleOp module) {
  SmallVector<::loom::MatmulOp> loomMatmuls;
  SmallVector<::loom::BatchMatmulOp> loomBatchMatmuls;
  module.walk([&](::loom::MatmulOp op) { loomMatmuls.push_back(op); });
  module.walk(
      [&](::loom::BatchMatmulOp op) { loomBatchMatmuls.push_back(op); });

  for (::loom::MatmulOp op : loomMatmuls) {
    OpBuilder builder(op);
    builder.create<linalg::MatmulOp>(
        op.getLoc(), ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{op.getOuts()});
    op.erase();
  }

  for (::loom::BatchMatmulOp op : loomBatchMatmuls) {
    OpBuilder builder(op);
    builder.create<linalg::BatchMatmulOp>(
        op.getLoc(), ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{op.getOuts()});
    op.erase();
  }

  return success();
}

static bool isComputeKernelFunc(func::FuncOp func) {
  if (auto threadAttr =
          func->getAttrOfType<ThreadTypeAttr>(ThreadTypeAttr::name)) {
    return threadAttr.getValue() == ThreadType::Compute;
  }
  return func.getName().ends_with("__compute");
}

static Value stripMemrefCasts(Value value) {
  Value current = value;
  while (auto cast = current.getDefiningOp<memref::CastOp>())
    current = cast.getSource();
  return current;
}

static FailureOr<ReduceProtocol>
parseReduceProtocolOption(StringRef optionValue) {
  std::string lowered = optionValue.trim().lower();
  StringRef value(lowered);
  if (value.empty() || value == "multi-slot")
    return ReduceProtocol::MultiSlot;
  if (value == "single-slot")
    return ReduceProtocol::SingleSlot;
  return failure();
}

class InsertMMInitPass
    : public PassWrapper<InsertMMInitPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertMMInitPass)

  StringRef getArgument() const override {
    return "loom-insert-mm-init";
  }

  StringRef getDescription() const override {
    return "Insert a single ttkernel.mm_init after parameter GetArgValOps in compute kernels";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    arith::ArithDialect, mlir::tt::ttkernel::TTKernelDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!isComputeKernelFunc(func))
        continue;

      bool hasMatmul = false;
      func.walk([&](linalg::MatmulOp) { hasMatmul = true; });
      if (!hasMatmul) {
        func.walk([&](linalg::BatchMatmulOp) { hasMatmul = true; });
      }
      if (!hasMatmul)
        continue;

      bool hasMMInit = false;
      func.walk([&](MatmulInitOp) { hasMMInit = true; });
      if (hasMMInit)
        continue;

      Block &entry = func.front();
      SmallVector<Value, 4> memrefCBs;
      struct MemrefArgCBInfo {
        BlockArgument arg;
        Value cb;
        bool isInput = false;
        bool isOutput = false;
      };
      SmallVector<MemrefArgCBInfo, 4> memrefInfos;
      Operation *lastGetArgVal = nullptr;
      for (Operation &op : entry) {
        auto getArgVal = dyn_cast<GetArgValOp>(op);
        if (!getArgVal)
          continue;
        lastGetArgVal = &op;
        if (isa<CBType>(getArgVal.getType()))
          memrefCBs.push_back(getArgVal.getResult());
      }

      unsigned memrefCbIndex = 0;
      for (BlockArgument arg : entry.getArguments()) {
        if (!isa<MemRefType, UnrankedMemRefType>(arg.getType()))
          continue;
        if (memrefCbIndex >= memrefCBs.size())
          break;
        memrefInfos.push_back({arg, memrefCBs[memrefCbIndex++]});
      }

      auto markRole = [&](BlockArgument arg, bool markInput) {
        for (MemrefArgCBInfo &info : memrefInfos) {
          if (info.arg != arg)
            continue;
          if (markInput)
            info.isInput = true;
          else
            info.isOutput = true;
          break;
        }
      };

      func.walk([&](::loom::CopyOp copyOp) {
        if (auto sourceRC =
                copyOp.getSource().getDefiningOp<memref::ReinterpretCastOp>()) {
          Value source = stripMemrefCasts(sourceRC.getSource());
          if (auto sourceArg = dyn_cast<BlockArgument>(source)) {
            if (sourceArg.getOwner() == &entry)
              markRole(sourceArg, /*markInput=*/true);
          }
        }

        if (auto destRC = copyOp.getDestination().getDefiningOp<memref::ReinterpretCastOp>()) {
          Value dest = stripMemrefCasts(destRC.getSource());
          if (auto destArg = dyn_cast<BlockArgument>(dest)) {
            if (destArg.getOwner() == &entry)
              markRole(destArg, /*markInput=*/false);
          }
        }
      });

      SmallVector<Value, 4> inputCBs;
      Value outputCB;
      for (const MemrefArgCBInfo &info : memrefInfos) {
        if (info.isInput)
          inputCBs.push_back(info.cb);
        if (!outputCB && info.isOutput)
          outputCB = info.cb;
      }

      if (inputCBs.size() < 2 || !outputCB) {
        func.emitError()
            << "failed to infer mm_init CB roles; expected two input DRAM "
               "memrefs and one output DRAM memref linked by loom.copy";
        signalPassFailure();
        return;
      }

      OpBuilder builder(func.getContext());
      if (lastGetArgVal)
        builder.setInsertionPointAfter(lastGetArgVal);
      else
        builder.setInsertionPointToStart(&entry);

      Value transpose = builder.create<arith::ConstantIntOp>(func.getLoc(), 0, 32);
      builder.create<MatmulInitOp>(func.getLoc(), inputCBs[0], inputCBs[1],
                                   outputCB, transpose);
    }
  }
};

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

  TileLoomToTTKernelPass()
      : matmulMergeBReaderIntoWriter(
            *this, "matmul-merge-b-reader-into-writer",
            llvm::cl::desc(
                "For linalg.matmul, keep the A reader on RISCV_1 and merge "
                "the B reader into the writer kernel on RISCV_0"),
            llvm::cl::init(false)),
        reduceProtocolOpt(
            *this, "reduce-protocol",
            llvm::cl::desc("Reduce synchronization protocol "
                           "(multi-slot|single-slot)"),
            llvm::cl::init("multi-slot")),
        reduceProtocolDeprecated(
            *this, "reduce-sum-protocol",
            llvm::cl::desc("[deprecated] Alias for reduce-protocol"),
            llvm::cl::init("")) {}

  TileLoomToTTKernelPass(const TileLoomToTTKernelPass &other)
      : PassWrapper(other),
        matmulMergeBReaderIntoWriter(
            *this, "matmul-merge-b-reader-into-writer",
            llvm::cl::desc(
                "For linalg.matmul, keep the A reader on RISCV_1 and merge "
                "the B reader into the writer kernel on RISCV_0"),
            llvm::cl::init(false)),
        reduceProtocolOpt(
            *this, "reduce-protocol",
            llvm::cl::desc("Reduce synchronization protocol "
                           "(multi-slot|single-slot)"),
            llvm::cl::init("multi-slot")),
        reduceProtocolDeprecated(
            *this, "reduce-sum-protocol",
            llvm::cl::desc("[deprecated] Alias for reduce-protocol"),
            llvm::cl::init("")) {
    matmulMergeBReaderIntoWriter =
        static_cast<bool>(other.matmulMergeBReaderIntoWriter);
    reduceProtocolOpt = other.reduceProtocolOpt;
    reduceProtocolDeprecated = other.reduceProtocolDeprecated;
  }

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

  Option<bool> matmulMergeBReaderIntoWriter;
  Option<std::string> reduceProtocolOpt;
  /// @deprecated Use reduce-protocol instead.
  Option<std::string> reduceProtocolDeprecated;

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

    if (failed(rewriteLoomLinearAlgebraToLinalg(module))) {
      signalPassFailure();
      return;
    }

    // Legacy reduce_sum is no longer supported in this lowering. The transport
    // path is now gather-only and sum math is handled by linalg.generic.
    bool sawLegacyReduceSum = false;
    module.walk([&](::loom::ReduceSumOp op) {
      if (sawLegacyReduceSum)
        return;
      sawLegacyReduceSum = true;
      op.emitOpError("is unsupported by loom-tileloom-to-ttkernel; migrate to "
                     "loom.gather + linalg.generic sum");
    });
    if (sawLegacyReduceSum) {
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
    if (matmulMergeBReaderIntoWriter) {
      if (failed(prepareMatmulBReaderMerge(module))) {
        signalPassFailure();
        return;
      }
    }

    if (failed(annotateVecLoadUsage(module))) {
      signalPassFailure();
      return;
    }

    specializeFunctionsForTTKernel(module);

    // Create shared compile-arg tracker for index management.
    auto compileArgTracker = std::make_shared<CompileArgTracker>();

    // Resolve reduce protocol option, preferring the deprecated alias when the
    // primary option was left at its default and the alias was explicitly set.
    std::string effectiveProtocolStr = reduceProtocolOpt;
    if (!reduceProtocolDeprecated.getValue().empty())
      effectiveProtocolStr = reduceProtocolDeprecated;

    FailureOr<ReduceProtocol> reduceProtocol =
        parseReduceProtocolOption(effectiveProtocolStr);
    if (failed(reduceProtocol)) {
      module.emitError() << "invalid reduce-protocol option: '"
                         << effectiveProtocolStr
                         << "' (expected 'multi-slot' or 'single-slot')";
      signalPassFailure();
      return;
    }

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

    if (failed(rewriteMatmulToBatch1Matmul(module))) {
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

    // Insert a single ttkernel.mm_init for compute kernels after all compile
    // parameters are materialized and before linalg.batch_matmul lowering.
    PassManager mmInitPm(context);
    mmInitPm.addPass(createInsertMMInitPass());
    if (failed(mmInitPm.run(module))) {
      signalPassFailure();
      return;
    }


    // Set up conversion target
    ConversionTarget target(*context);
    
    // Mark loom semaphore/copy ops as illegal in the main lowering stage.
    // loom.alloc is cleaned up in a dedicated follow-up conversion pass once
    // semaphore/copy rewrites have consumed it.
    target.addIllegalOp<::loom::SemaphoreTakeOp>();
    target.addIllegalOp<::loom::SemaphoreGiveOp>();
    target.addIllegalOp<::loom::CopyOp>();
    target.addIllegalOp<::loom::GatherOp>();
    target.addIllegalOp<::loom::ReduceSumOp>();
    
    // Mark memref operations that don't need conversion as legal
    // (they will be type-converted automatically)
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect>();
    // Normalize unsigned index div/mod before handing IR to EmitC lowering.
    target.addDynamicallyLegalOp<arith::DivUIOp>(
        [](arith::DivUIOp op) { return !op.getType().isIndex(); });
    target.addDynamicallyLegalOp<arith::RemUIOp>(
        [](arith::RemUIOp op) { return !op.getType().isIndex(); });
    target.addDynamicallyLegalOp<arith::CeilDivUIOp>(
        [](arith::CeilDivUIOp op) { return !op.getType().isIndex(); });
    target.addDynamicallyLegalOp<arith::CeilDivSIOp>(
        [](arith::CeilDivSIOp op) { return !op.getType().isIndex(); });

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

    // linalg dialect is generally legal, but we require conversion for
    // matmul-style ops so they can be lowered to TTKernel matmul blocks.
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
    target.addDynamicallyLegalOp<linalg::TransposeOp>(
        [&](linalg::TransposeOp op) {
          return !mlir::loom::shouldConvertComputeLinalgTranspose(op);
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
    target.addDynamicallyLegalOp<memref::ExpandShapeOp>(
        [&](memref::ExpandShapeOp op) {
          (void)op;
          return false;
        });
    target.addLegalDialect<mlir::tt::ttkernel::TTKernelDialect,
                           mlir::emitc::EmitCDialect>();

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
                                       compileArgTracker, *reduceProtocol);
    // Add compute operation conversion patterns (e.g., linalg.matmul)
    populateComputeOpConversionPatterns(patterns, typeConverter, context,
                                        compileArgTracker, *reduceProtocol);

    // Apply conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Host specialization can leave loop shells containing only unused
    // get_arg_val/index_cast artifacts. Remove them before final cleanup.
    eraseHostLoweringArtifacts(module);
    eraseDeadRuntimeSetupArtifacts(module);

    // Finalize loom.alloc cleanup after all dependent rewrites have run.
    ConversionTarget allocCleanupTarget(*context);
    allocCleanupTarget.markUnknownOpDynamicallyLegal(
        [](Operation *) { return true; });
    allocCleanupTarget.addIllegalOp<::loom::AllocOp>();

    RewritePatternSet allocCleanupPatterns(context);
    populateLoomAllocCleanupPatterns(allocCleanupPatterns, typeConverter,
                                     context);

    if (failed(applyPartialConversion(module, allocCleanupTarget,
                                      std::move(allocCleanupPatterns)))) {
      signalPassFailure();
      return;
    }

    // Mapping metadata on loop shells is no longer needed after lowering and
    // leaks unregistered Loom attrs into TT-facing pipelines.
    eraseLoomLoopAttrs(module);

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
    postPm.addPass(createSymbolDCEPass());
    if (failed(postPm.run(module))) {
      signalPassFailure();
      return;
    }

    // Final stage: strip descriptor dialect ops (df/adl) from the module.
    eraseDescriptorDialectOps(module);
  }
};

/**
 * @brief Post-EmitC host signature rewrite pass.
 *
 * @details Rewrites each generated host helper signature to:
 *          - `__host_cpp`: one `std::vector<bfloat16>&` per original memref
 *            argument
 *          - `__host_pybind`: one `const ttnn::Tensor&` per original memref
 *            argument
 *          - both variants then receive core-range args
 *          - only `__host_cpp` receives a trailing `IDevice*`
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
    return "Rewrite host_cpp/host_pybind signatures to typed memref inputs and core range args, with IDevice* only on host_cpp";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, emitc::EmitCDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    Type hostVectorType =
        emitc::OpaqueType::get(ctx, "std::vector<bfloat16>&");
    Type hostTensorType =
        emitc::OpaqueType::get(ctx, "const ttnn::Tensor&");
    Type coreCoordType = emitc::OpaqueType::get(ctx, "uint32_t");

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      bool isHostCpp =
          func.getName().ends_with("__host_cpp") ||
          func.getName().ends_with("__host");
      bool isHostPybind = func.getName().ends_with("__host_pybind");
      if (!isHostCpp && !isHostPybind)
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
        newInputs.push_back(isHostPybind ? hostTensorType : hostVectorType);
      newInputs.push_back(coreCoordType); // start_core_x
      newInputs.push_back(coreCoordType); // start_core_y
      newInputs.push_back(coreCoordType); // end_core_x
      newInputs.push_back(coreCoordType); // end_core_y
      if (!isHostPybind)
        newInputs.push_back(
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "IDevice")));

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

std::unique_ptr<Pass> mlir::loom::createInsertMMInitPass() {
  return std::make_unique<InsertMMInitPass>();
}

void mlir::loom::registerTileLoomToTTKernelPass() {
  PassRegistration<InsertMMInitPass>();
  PassRegistration<TileLoomToTTKernelPass>();
  PassRegistration<PostEmitCHostSignaturePass>();
}
