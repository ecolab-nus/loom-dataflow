#include "reinterpret_cast_reuse.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;

namespace {

static bool dependsOn(Value value, Value target) {
  if (!value)
    return false;
  if (value == target)
    return true;

  SmallPtrSet<Value, 16> visited;
  SmallVector<Value, 16> worklist;
  worklist.push_back(value);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (current == target)
      return true;

    if (auto barg = llvm::dyn_cast<BlockArgument>(current)) {
      (void)barg;
      continue;
    }

    Operation *def = current.getDefiningOp();
    if (!def)
      continue;

    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }

  return false;
}

class AnnotateReinterpretCastReusePass
    : public PassWrapper<AnnotateReinterpretCastReusePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AnnotateReinterpretCastReusePass)

  StringRef getArgument() const override {
    return "tmd-annotate-reinterpret-cast-reuse";
  }
  StringRef getDescription() const override {
    return "Annotate memref.reinterpret_cast ops with iterator reuse info";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    IntegerType i64Type = IntegerType::get(ctx, 64);
    OpPrintingFlags printingFlags;
    printingFlags.useLocalScope();
    AsmState asmState(module, printingFlags);

    // The pass tags every `memref.reinterpret_cast` with a `tmd.reuse`
    // dictionary summarising how its offset varies with surrounding loop
    // iterators. The dictionary is grouped by iterator class:
    //   - `spatial`    : innermost-to-outermost `affine.parallel` loops.
    //   - `temporal`   : surrounding `affine.for` loops (waves).
    //   - `sequential` : surrounding `scf.for` loops (per-core tiles).
    // Each entry <dict> contains:
    //   * `iterator`   – SSA name of the induction variable (e.g. `%arg13`).
    //   * `depth`      – integer depth (0 = outermost iterator touching the
    //                    op).
    //   * `reuse_type` – `no_reuse` if the offset changes with that iterator,
    //                    `total_reuse` if it stays constant (partial reuse is
    //                    reserved for future refinements).
    //   * `volume`     – amount of data reused for the iterator. Currently 0
    //                    for `no_reuse` and the full block size for
    //                    `total_reuse`; values become -1 when the block shape
    //                    is dynamic.
    //   * `mapped_to` – spatial dimension name taken from `tmd.mapped_to`.
    module.walk([&](memref::ReinterpretCastOp op) {
      SmallVector<Value, 4> dynamicOffsets;
      for (OpFoldResult ofr : op.getMixedOffsets())
        if (auto val = ofr.dyn_cast<Value>())
          dynamicOffsets.push_back(val);

      auto blockVolumeBytes = [&]() -> std::optional<int64_t> {
        auto resultType = llvm::dyn_cast<MemRefType>(op.getResult().getType());
        if (!resultType)
          return std::nullopt;
        if (!resultType.hasStaticShape())
          return std::nullopt;
        int64_t numElements = resultType.getNumElements();
        Type elemType = resultType.getElementType();
        int64_t elemBits = 0;
        if (auto intTy = llvm::dyn_cast<IntegerType>(elemType))
          elemBits = intTy.getWidth();
        else if (auto floatTy = llvm::dyn_cast<FloatType>(elemType))
          elemBits = floatTy.getWidth();
        else
          return std::nullopt;
        if (elemBits % 8 != 0)
          return std::nullopt;
        return numElements * (elemBits / 8);
      }();

      SmallVector<Operation *, 8> loopStack;
      for (Operation *parent = op->getParentOp(); parent;
           parent = parent->getParentOp()) {
        if (isa<affine::AffineParallelOp, affine::AffineForOp, scf::ForOp>(
                parent))
          loopStack.push_back(parent);
      }
      if (loopStack.empty())
        return;

      std::reverse(loopStack.begin(), loopStack.end());

      SmallVector<Attribute, 4> spatialEntries;
      SmallVector<Attribute, 4> temporalEntries;
      SmallVector<Attribute, 4> sequentialEntries;

      auto annotateIterator = [&](Value iv, SmallVectorImpl<Attribute> &bucket,
                                  StringAttr mappedTo, unsigned depth) {
        bool variant = llvm::any_of(dynamicOffsets, [&](Value v) {
          return dependsOn(v, iv);
        });
        StringRef reuseType = variant ? "no_reuse" : "total_reuse";
        int64_t reuseVolume = 0;
        if (!variant) {
          if (blockVolumeBytes)
            reuseVolume = *blockVolumeBytes;
          else
            reuseVolume = -1; // unknown block size
        }
        NamedAttrList attrs;
        std::string iteratorName;
        {
          llvm::raw_string_ostream os(iteratorName);
          iv.printAsOperand(os, asmState);
        }
        if (iteratorName.empty())
          iteratorName = "<unnamed>";
        attrs.append("iterator", StringAttr::get(ctx, iteratorName));
        attrs.append("depth", IntegerAttr::get(i64Type, depth));
        attrs.append("reuse_type", StringAttr::get(ctx, reuseType));
        attrs.append("volume", IntegerAttr::get(i64Type, reuseVolume));
        if (mappedTo)
          attrs.append("mapped_to", mappedTo);
        bucket.push_back(DictionaryAttr::get(ctx, attrs));
      };

      for (auto [depth, loop] : llvm::enumerate(loopStack)) {
        if (auto par = dyn_cast<affine::AffineParallelOp>(loop)) {
          StringAttr mapped =
              loop->getAttrOfType<StringAttr>("tmd.mapped_to");
          for (Value iv : par.getIVs())
            annotateIterator(iv, spatialEntries, mapped, depth);
        } else if (auto affFor = dyn_cast<affine::AffineForOp>(loop)) {
          annotateIterator(affFor.getInductionVar(), temporalEntries,
                           StringAttr(), depth);
        } else if (auto scfFor = dyn_cast<scf::ForOp>(loop)) {
          annotateIterator(scfFor.getInductionVar(), sequentialEntries,
                           StringAttr(), depth);
        }
      }

      NamedAttrList reuseAttrs;
      if (!spatialEntries.empty())
        reuseAttrs.append("spatial", ArrayAttr::get(ctx, spatialEntries));
      if (!temporalEntries.empty())
        reuseAttrs.append("temporal", ArrayAttr::get(ctx, temporalEntries));
      if (!sequentialEntries.empty())
        reuseAttrs.append("sequential",
                          ArrayAttr::get(ctx, sequentialEntries));

      if (!reuseAttrs.empty())
        op->setAttr("tmd.reuse", DictionaryAttr::get(ctx, reuseAttrs));
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
tmd::passes::createAnnotateReinterpretCastReusePass() {
  return std::make_unique<AnnotateReinterpretCastReusePass>();
}

void tmd::passes::registerAnnotateReinterpretCastReusePass() {
  PassRegistration<AnnotateReinterpretCastReusePass>();
}
