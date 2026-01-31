/**
 * @file analyze_reuse.cpp
 * @brief Implementation: update reuse attributes of `loom.view`.
 * @details
 * Strategy
 * - For each `loom.view`, collect its dynamic offsets.
 * - Walk outward to gather enclosing loops in lexical order, distinguishing
 *   `affine.parallel` (spatial), `affine.for` (temporal), and `scf.for`
 *   (sequential).
 * - For each iterator type, check whether any dynamic offset depends on the
 *   iterators of that type. If at least one iterator of a type has no
 * dependency (has reuse), set the corresponding reuse attribute to true.
 * - Directly update the `sequential_reuse`, `spatial_reuse`, and
 * `temporal_reuse` attributes of the operation.
 */

#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

// Loom dialect headers
#include "LoomDialect.h.inc"
#include "LoomEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LoomTypes.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

/// @brief Check whether `value` depends (transitively) on `target`.
/// @details Walks the SSA def-use graph backward from `value` to determine if
/// `target` appears among its transitive operands. Block arguments stop the
/// walk.
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

class AnnotateViewReusePass
    : public PassWrapper<AnnotateViewReusePass, OperationPass<ModuleOp>> {
public:
  /**
   * @brief Update reuse attributes of `loom.view` operations.
   *
   * @details For each view, the pass identifies surrounding iterator variables
   * (spatial `affine.parallel`, temporal `affine.for`, sequential `scf.for`),
   * determines whether the view's dynamic offsets vary with iterators of each
   * type, and directly updates the `sequential_reuse`, `spatial_reuse`, and
   * `temporal_reuse` attributes.
   */
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotateViewReusePass)

  /// Command-line flag name.
  StringRef getArgument() const override { return "loom-annotate-view-reuse"; }
  /// Short pass description for diagnostics and help.
  StringRef getDescription() const override {
    return "Update reuse attributes of loom.view ops";
  }

  /**
   * @brief Execute the pass over the module.
   *
   * @details Walk all `loom.view` ops, collect dynamic offsets,
   * derive enclosing iterators, and update reuse attributes based on whether
   * offsets depend on iterators of each type.
   *
   * Note: This pass operates directly on the existing module without creating
   * a new one, so module-level attributes are automatically preserved.
   */
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    module.walk([&](Operation *op) {
      // Check if this is a loom.view operation
      if (op->getName().getStringRef() != "loom.view")
        return;

      auto viewOp = dyn_cast<loom::ViewOp>(op);
      if (!viewOp)
        return;

      // Collect dynamic offsets
      SmallVector<Value, 4> dynamicOffsets;
      for (Value offsetVal : viewOp.getOffsets()) {
        dynamicOffsets.push_back(offsetVal);
      }

      // Collect enclosing loops in lexical order (outermost to innermost).
      SmallVector<Operation *, 8> loopStack;
      for (Operation *parent = op->getParentOp(); parent;
           parent = parent->getParentOp()) {
        if (isa<affine::AffineParallelOp, affine::AffineForOp, scf::ForOp>(
                parent))
          loopStack.push_back(parent);
      }
      if (loopStack.empty())
        return;

      // Track reuse status for each iterator type.
      bool hasSpatialReuse = false;
      bool hasTemporalReuse = false;
      bool hasSequentialReuse = false;

      // Check each loop for reuse based on its loom.iter_type attribute.
      for (Operation *loop : loopStack) {
        auto iterTypeAttr =
            loop->getAttrOfType<loom::IterTypeAttr>("loom.iter_type");
        if (!iterTypeAttr)
          continue;

        loom::IterType iterType = iterTypeAttr.getValue();

        // Get induction variables (IVs) for the loop.
        SmallVector<Value, 4> ivs;
        if (auto par = dyn_cast<affine::AffineParallelOp>(loop)) {
          for (Value iv : par.getIVs())
            ivs.push_back(iv);
        } else if (auto affFor = dyn_cast<affine::AffineForOp>(loop)) {
          ivs.push_back(affFor.getInductionVar());
        } else if (auto scfFor = dyn_cast<scf::ForOp>(loop)) {
          ivs.push_back(scfFor.getInductionVar());
        }

        // Check if any induction variable has reuse (i.e., offsets don't depend
        // on it).
        bool loopHasReuse = false;
        for (Value iv : ivs) {
          bool isVariant = llvm::any_of(
              dynamicOffsets, [&](Value v) { return dependsOn(v, iv); });
          if (!isVariant) {
            loopHasReuse = true;
            break;
          }
        }

        if (loopHasReuse) {
          if (iterType == loom::IterType::Spatial)
            hasSpatialReuse = true;
          else if (iterType == loom::IterType::Temporal)
            hasTemporalReuse = true;
          else if (iterType == loom::IterType::Sequential)
            hasSequentialReuse = true;
        }
      }

      // Update the reuse attributes directly on the operation.
      op->setAttr("sequential_reuse", BoolAttr::get(ctx, hasSequentialReuse));
      op->setAttr("spatial_reuse", BoolAttr::get(ctx, hasSpatialReuse));
      op->setAttr("temporal_reuse", BoolAttr::get(ctx, hasTemporalReuse));
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> loom::passes::createAnnotateViewReusePass() {
  /**
   * @brief Create the view reuse annotation pass.
   */
  return std::make_unique<AnnotateViewReusePass>();
}
