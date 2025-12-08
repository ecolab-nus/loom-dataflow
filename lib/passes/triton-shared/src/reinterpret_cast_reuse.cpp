/**
 * @file reinterpret_cast_reuse.cpp
 * @brief Implementation: annotate `memref.reinterpret_cast` with `loom.reuse`.
 * @details
 * Strategy
 * - For each `memref.reinterpret_cast`, collect its dynamic offsets.
 * - Walk outward to gather enclosing loops in lexical order, distinguishing
 *   `affine.parallel` (spatial), `affine.for` (temporal), and `scf.for`
 *   (sequential).
 * - For each IV, check whether any dynamic offset depends on the IV via a
 *   dependency walk. If yes, mark `reuse_type = no_reuse`; otherwise
 *   `total_reuse` and estimate `volume` by the result memref block size in
 *   bytes (or -1 when unknown).
 * - Emit `loom.reuse` as a nested dictionary with three arrays: `spatial`,
 *   `temporal`, `sequential`. Each element carries `iterator`, `depth`,
 *   `reuse_type`, `volume`, and optional `mapped_to`.
 */

#include "reinterpret_cast_reuse.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;

namespace {

/// @brief Check whether `value` depends (transitively) on `target`.
/// @details Walks the SSA def-use graph backward from `value` to determine if
/// `target` appears among its transitive operands. Block arguments stop the walk.
static bool dependsOn(Value value, Value target) {
  if (!value || value == target)
    return value == target;

  SmallPtrSet<Value, 16> visited;
  SmallVector<Value, 16> worklist = {value};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (current == target)
      return true;

    // Block arguments stop the walk (treated as leaves).
    if (llvm::isa<BlockArgument>(current))
      continue;

    if (Operation *def = current.getDefiningOp()) {
      worklist.append(def->operand_begin(), def->operand_end());
    }
  }

  return false;
}

class AnnotateReinterpretCastReusePass
    : public PassWrapper<AnnotateReinterpretCastReusePass,
                         OperationPass<ModuleOp>> {
public:
  /**
   * @brief Annotate `memref.reinterpret_cast` with iterator reuse metadata.
   *
   * @details For each cast, the pass identifies surrounding iterator variables
   * (spatial `affine.parallel`, temporal `affine.for`, sequential `scf.for`),
   * determines whether the cast's dynamic offsets vary with each iterator, and
   * attaches the summary as a `loom.reuse` dictionary grouped by iterator
   * classes.
   */
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotateReinterpretCastReusePass)

  /// Command-line flag name.
  StringRef getArgument() const override {
    return "loom-annotate-reinterpret-cast-reuse";
  }
  /// Short pass description for diagnostics and help.
  StringRef getDescription() const override {
    return "Annotate memref.reinterpret_cast ops with iterator reuse info";
  }

  /**
   * @brief Execute the annotation pass over the module.
   *
   * @details Walk all `memref.reinterpret_cast` ops, collect dynamic offsets,
   * derive enclosing iterators, classify reuse (`no_reuse` vs `total_reuse`),
   * estimate reuse volume (bytes or -1), and attach a `loom.reuse` dictionary
   * with arrays for `spatial`, `temporal`, and `sequential` iterators.
   */
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    IntegerType i64Type = IntegerType::get(ctx, 64);
    OpPrintingFlags printingFlags;
    printingFlags.useLocalScope();
    AsmState asmState(module, printingFlags);

    // The pass tags every `memref.reinterpret_cast` with a `loom.reuse`
    // dictionary summarising how its offset varies with surrounding loop
    // iterators. The dictionary is grouped by iterator class:
    //   - `spatial`    : innermost-to-outermost `affine.parallel` loops.
    //   - `temporal`   : surrounding `affine.for` loops (waves).
    //   - `sequential` : surrounding `scf.for` loops (per-core tiles).
    // Each entry <dict> contains:
    //   * `iterator`   – SSA name of the induction variable (e.g. `%arg13`).
    //   * `depth`      – integer depth (0 = the outermost enclosing loop,
    //                    increasing by one as we move inward toward the
    //                    `memref.reinterpret_cast`). This is computed by
    //                    climbing the parent chain from the cast outwards,
    //                    reversing the stack, and numbering the loops so the
    //                    entry reflects how far the iterator sits in the nest.
    //   * `reuse_type` – `no_reuse` if the offset changes with that iterator,
    //                    `total_reuse` if it stays constant (partial reuse is
    //                    reserved for future refinements).
    //   * `volume`     – amount of data reused for the iterator. Currently 0
    //                    for `no_reuse` and the full block size for
    //                    `total_reuse`; values become -1 when the block shape
    //                    is dynamic.
    //   * `mapped_to` – spatial dimension name taken from `loom.mapped_to`.
    module.walk([&](memref::ReinterpretCastOp op) {
      // Collect dynamic offsets (non-constant ones).
      SmallVector<Value, 4> dynamicOffsets;
      for (OpFoldResult ofr : op.getMixedOffsets()) {
        if (auto val = ofr.dyn_cast<Value>())
          dynamicOffsets.push_back(val);
      }

      // Compute block volume in bytes, or nullopt if not statically determinable.
      auto blockVolumeBytes = [&]() -> std::optional<int64_t> {
        auto resultType = llvm::dyn_cast<MemRefType>(op.getResult().getType());
        if (!resultType || !resultType.hasStaticShape())
          return std::nullopt;
        
        int64_t numElements = resultType.getNumElements();
        Type elemType = resultType.getElementType();
        
        int64_t elemBits = 0;
        if (auto intTy = llvm::dyn_cast<IntegerType>(elemType)) {
          elemBits = intTy.getWidth();
        } else if (auto floatTy = llvm::dyn_cast<FloatType>(elemType)) {
          elemBits = floatTy.getWidth();
        } else {
          return std::nullopt;
        }
        
        if (elemBits % 8 != 0)
          return std::nullopt;
        
        return numElements * (elemBits / 8);
      }();

      // Collect enclosing loops in lexical order (outermost to innermost).
      SmallVector<Operation *, 8> loopStack;
      for (Operation *parent = op->getParentOp(); parent;
           parent = parent->getParentOp()) {
        if (isa<affine::AffineParallelOp, affine::AffineForOp, scf::ForOp>(parent))
          loopStack.push_back(parent);
      }
      if (loopStack.empty())
        return;
      
      // Reverse to get outermost-to-innermost order for depth calculation.
      std::reverse(loopStack.begin(), loopStack.end());

      SmallVector<Attribute, 4> spatialEntries;
      SmallVector<Attribute, 4> temporalEntries;
      SmallVector<Attribute, 4> sequentialEntries;

      auto annotateIterator = [&](Value iv, SmallVectorImpl<Attribute> &bucket,
                                  StringAttr mappedTo, unsigned depth) {
        // Check if any dynamic offset depends on this iterator.
        bool hasVariant = llvm::any_of(dynamicOffsets,
                                      [&](Value v) { return dependsOn(v, iv); });
        
        // Compute reuse volume: 0 for variant, block size for total reuse, -1 if unknown.
        int64_t reuseVolume = hasVariant ? 0 : 
                             (blockVolumeBytes ? *blockVolumeBytes : -1);
        
        // Get iterator name for display.
        std::string iteratorName;
        {
          llvm::raw_string_ostream os(iteratorName);
          iv.printAsOperand(os, asmState);
        }
        
        NamedAttrList attrs;
        attrs.append("iterator", StringAttr::get(ctx, 
                     iteratorName.empty() ? "<unnamed>" : iteratorName));
        attrs.append("depth", IntegerAttr::get(i64Type, depth));
        attrs.append("reuse_type", StringAttr::get(ctx, 
                     hasVariant ? "no_reuse" : "total_reuse"));
        attrs.append("volume", IntegerAttr::get(i64Type, reuseVolume));
        if (mappedTo)
          attrs.append("mapped_to", mappedTo);
        
        bucket.push_back(DictionaryAttr::get(ctx, attrs));
      };

      for (auto [depth, loop] : llvm::enumerate(loopStack)) {
        if (auto par = dyn_cast<affine::AffineParallelOp>(loop)) {
          auto mapped = loop->getAttrOfType<StringAttr>("loom.mapped_to");
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

      // Build reuse dictionary with non-empty entry arrays.
      NamedAttrList reuseAttrs;
      if (!spatialEntries.empty())
        reuseAttrs.append("spatial", ArrayAttr::get(ctx, spatialEntries));
      if (!temporalEntries.empty())
        reuseAttrs.append("temporal", ArrayAttr::get(ctx, temporalEntries));
      if (!sequentialEntries.empty())
        reuseAttrs.append("sequential", ArrayAttr::get(ctx, sequentialEntries));

      if (!reuseAttrs.empty())
        op->setAttr("loom.reuse", DictionaryAttr::get(ctx, reuseAttrs));
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createAnnotateReinterpretCastReusePass() {
  /**
   * @brief Create the reinterpret_cast reuse annotation pass.
   */
  return std::make_unique<AnnotateReinterpretCastReusePass>();
}

void loom::passes::registerAnnotateReinterpretCastReusePass() {
  /**
   * @brief Register the reinterpret_cast reuse annotation pass.
   */
  PassRegistration<AnnotateReinterpretCastReusePass>();
}
