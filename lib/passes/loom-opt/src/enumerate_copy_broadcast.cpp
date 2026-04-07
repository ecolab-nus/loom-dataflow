/**
 * @file enumerate_copy_broadcast.cpp
 * @brief Implementation for enumerating copy broadcast choices.
 * @details
 * This pass analyzes loom.copy operations and checks their source operations
 * for spatial reuse information from loom.subview operations.
 * It enumerates all possible broadcast choices (no broadcast, broadcast on dim_x,
 * broadcast on dim_y, broadcast on both) and generates function clones for each
 * combination.
 *
 * Hardware dimension sizes are read from adl.spatial_dim operations in the
 * outer module. No df dialect is used.
 */

#include "Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "utils.h"
#include "llvm/ADT/SmallVector.h"

// Include Loom dialect headers for CopyOp and SubviewOp
#include "mlir/Interfaces/ViewLikeInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

// Include ADL dialect headers for adl.spatial_dim
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
#define GET_OP_CLASSES
#include "ADLOps.h.inc"

using namespace mlir;

namespace {

/**
 * @brief A single broadcast choice for a loom.copy operation.
 * @details Records the broadcast values to apply and a short label used in
 * generated function names.
 */
struct BroadcastChoice {
  SmallVector<int64_t> values; // e.g., {1, 1}, {1, 8}, {8, 1}, {8, 8}
  std::string label;           // "n" (none), "dim_y", "dim_x", "a" (all)
};

/**
 * @brief Check whether a value depends (transitively) on a target value.
 * @details Walks the SSA def-use graph backward from value to determine if
 * target appears among its transitive operands. Block arguments stop the walk.
 * @param value The value to check for dependency.
 * @param target The target value to search for.
 * @return true if value depends on target, false otherwise.
 */
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

/**
 * @brief Set the broadcast attribute on a loom.copy operation.
 * @details Creates an ArrayAttr of IntegerAttr from the broadcast values and
 * sets it on the operation.
 * @param copyOp The copy operation to set the attribute on.
 * @param broadcastValues The broadcast values to set.
 */
static void setBroadcastAttribute(Operation *copyOp,
                                  ArrayRef<int64_t> broadcastValues) {
  MLIRContext *ctx = copyOp->getContext();
  SmallVector<Attribute> broadcastAttrs;
  for (int64_t val : broadcastValues) {
    broadcastAttrs.push_back(IntegerAttr::get(IntegerType::get(ctx, 64), val));
  }
  auto finalBroadcastAttr = ArrayAttr::get(ctx, broadcastAttrs);
  copyOp->setAttr("broadcast", finalBroadcastAttr);
}

/**
 * @brief Collect all enclosing affine.parallel loops for an operation.
 * @details Walks up the parent chain to find all affine.parallel operations
 * that enclose the given operation.
 * @param op The operation to find enclosing loops for.
 * @return A vector of enclosing affine.parallel operations in order from
 * outermost to innermost.
 */
static SmallVector<affine::AffineParallelOp>
collectEnclosingParallelLoops(Operation *op) {
  SmallVector<affine::AffineParallelOp> parallelLoops;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto par = dyn_cast<affine::AffineParallelOp>(parent))
      parallelLoops.push_back(par);
  }
  return parallelLoops;
}

/**
 * @brief Check if any offset value depends on any of the given induction
 * variables.
 * @param offsets The offset values to check.
 * @param ivs The induction variables to check dependency against.
 * @return true if any offset depends on any IV, false otherwise.
 */
static bool checkOffsetDependencyOnIVs(ArrayRef<Value> offsets,
                                       ValueRange ivs) {
  for (Value iv : ivs) {
    for (Value offset : offsets) {
      if (dependsOn(offset, iv)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * @brief Find and verify the subview source operation for a copy operation.
 * @details Checks if the copy operation's source operand is defined by a
 * loom.subview with spatial_reuse enabled.
 * @param copyOp The loom.copy operation to analyze.
 * @return The subview operation if found and valid, nullptr otherwise.
 */
static loom::SubviewOp findSubviewSource(loom::CopyOp copyOp) {
  if (!copyOp)
    return nullptr;

  Value source = copyOp.getSource();
  auto subviewOp = source.getDefiningOp<loom::SubviewOp>();
  if (!subviewOp)
    return nullptr;

  if (!subviewOp.getSpatialReuse())
    return nullptr;

  return subviewOp;
}

/**
 * @brief Get the size of a named adl.spatial_dim from the outer module.
 * @details Walks all adl::SpatialDimOp operations in the module looking for
 * one with the given symbol name.
 * @param outerModule The module to search.
 * @param symName The symbol name to look up (e.g., "dim_x" or "dim_y").
 * @return The size if found, or std::nullopt if not found.
 */
static std::optional<uint64_t>
getSpatialDimSizeFromADL(ModuleOp outerModule, StringRef symName) {
  std::optional<uint64_t> result;
  outerModule.walk([&](adl::SpatialDimOp dimOp) {
    if (dimOp.getSymName() == symName)
      result = dimOp.getSize();
  });
  return result;
}

/**
 * @brief Find all broadcast candidate choices for a loom.copy operation.
 * @details Analyzes the copy's source subview for spatial reuse, then checks
 * which enclosing spatial parallel loops' IVs the subview offsets do NOT
 * depend on. Under the 2D assumption (dims @dim_x and @dim_y):
 *   - Offset independent of @dim_x IV → can broadcast on @dim_y → {1, size_y}
 *   - Offset independent of @dim_y IV → can broadcast on @dim_x → {size_x, 1}
 *   - Both independent → also add {size_x, size_y}
 * Always includes the no-broadcast choice {1, 1}.
 * @param copyOp The loom.copy operation to analyze.
 * @param outerModule The top-level module containing adl.spatial_dim ops.
 * @return A vector of BroadcastChoice candidates (always non-empty).
 */
static SmallVector<BroadcastChoice>
findCopyBroadcastCandidates(loom::CopyOp copyOp, ModuleOp outerModule) {
  SmallVector<BroadcastChoice> candidates;
  candidates.push_back({{1, 1}, "n"}); // Always include no-broadcast option

  auto subviewOp = findSubviewSource(copyOp);
  if (!subviewOp)
    return candidates;

  SmallVector<Value> offsets(subviewOp.getOffsets().begin(),
                             subviewOp.getOffsets().end());
  if (offsets.empty())
    return candidates;

  SmallVector<affine::AffineParallelOp> parallelLoops =
      collectEnclosingParallelLoops(copyOp);

  // 2D assumption: track @dim_x and @dim_y independence and sizes
  bool xIndependent = false;
  bool yIndependent = false;
  int64_t sizeX = 0;
  int64_t sizeY = 0;

  for (auto par : parallelLoops) {
    auto mappedToAttr = par->getAttrOfType<SymbolRefAttr>("loom.physical_dim");
    if (!mappedToAttr)
      continue;

    StringRef dimName = mappedToAttr.getRootReference().getValue();
    bool independent = !checkOffsetDependencyOnIVs(offsets, par.getIVs());

    auto sizeOpt = getSpatialDimSizeFromADL(outerModule, dimName);
    if (!sizeOpt)
      continue;

    if (dimName == "dim_x") {
      xIndependent = independent;
      sizeX = static_cast<int64_t>(sizeOpt.value());
    } else if (dimName == "dim_y") {
      yIndependent = independent;
      sizeY = static_cast<int64_t>(sizeOpt.value());
    }
  }

  bool canBroadcastOnY = xIndependent && sizeY > 0;
  bool canBroadcastOnX = yIndependent && sizeX > 0;

  if (canBroadcastOnY && canBroadcastOnX) {
    candidates.push_back({{1, sizeY}, "dim_y"});
    candidates.push_back({{sizeX, 1}, "dim_x"});
    candidates.push_back({{sizeX, sizeY}, "a"});
  } else if (canBroadcastOnY) {
    candidates.push_back({{1, sizeY}, "dim_y"});
  } else if (canBroadcastOnX) {
    candidates.push_back({{sizeX, 1}, "dim_x"});
  }

  return candidates;
}

/**
 * @brief Generate a function name based on broadcast choices.
 * @param baseName The original function name.
 * @param choices The broadcast choices, one per copy op in the function.
 * @return The generated function name.
 */
static std::string generateFunctionName(StringRef baseName,
                                        ArrayRef<BroadcastChoice> choices) {
  std::string newName = baseName.str() + "__";
  for (size_t i = 0; i < choices.size(); ++i) {
    if (i > 0)
      newName += "_";
    newName += choices[i].label;
  }
  return newName;
}

/**
 * @brief Recursively generate all Cartesian product combinations of broadcast
 * choices.
 */
static void generateCartesianProduct(
    const SmallVector<SmallVector<BroadcastChoice>> &allCandidates,
    size_t depth, SmallVector<BroadcastChoice> &current,
    SmallVector<SmallVector<BroadcastChoice>> &results) {
  if (depth == allCandidates.size()) {
    results.push_back(current);
    return;
  }

  for (const auto &choice : allCandidates[depth]) {
    current.push_back(choice);
    generateCartesianProduct(allCandidates, depth + 1, current, results);
    current.pop_back();
  }
}

/**
 * @brief Find all loom.copy operations in a function.
 * @param func The function to search.
 * @return A vector of loom.copy operations found in the function.
 */
static SmallVector<loom::CopyOp> findCopyOpsInFunc(func::FuncOp func) {
  SmallVector<loom::CopyOp> copyOps;
  func.walk([&](loom::CopyOp copyOp) { copyOps.push_back(copyOp); });
  return copyOps;
}

/**
 * @brief Class responsible for enumerating and generating function clones
 * based on copy operation broadcast choices.
 */
class CopyBroadcastEnumerator {
public:
  CopyBroadcastEnumerator(ModuleOp module)
      : module(module), builder(module.getBodyRegion()) {}

  /// @brief Enumerate choices for all functions in the module.
  void enumerate() {
    SmallVector<func::FuncOp> funcs = loom::utils::collectFunctions(module);
    for (func::FuncOp func : funcs) {
      processFunction(func);
    }
  }

private:
  void processFunction(func::FuncOp originalFunc) {
    ModuleOp parentModule = loom::utils::getParentModule(originalFunc);
    DictionaryAttr moduleAttrs =
        parentModule ? parentModule->getAttrDictionary() : nullptr;

    auto copyOps = findCopyOpsInFunc(originalFunc);
    if (copyOps.empty())
      return;

    // Collect broadcast candidates for each copy operation
    SmallVector<SmallVector<BroadcastChoice>> allCandidates;
    for (auto copyOp : copyOps) {
      allCandidates.push_back(findCopyBroadcastCandidates(copyOp, module));
    }

    // Generate all Cartesian product combinations
    SmallVector<SmallVector<BroadcastChoice>> combinations;
    SmallVector<BroadcastChoice> current;
    generateCartesianProduct(allCandidates, 0, current, combinations);

    Operation *insertAfter = parentModule;
    for (const auto &combo : combinations) {
      std::string newName =
          generateFunctionName(originalFunc.getSymName(), combo);

      func::FuncOp clonedFunc = loom::utils::cloneFuncWithConstraints(
          builder, originalFunc, newName, moduleAttrs, "EnumerateCopyBroadcast",
          [&](func::FuncOp func) { return applyChoices(func, combo); },
          insertAfter);

      if (clonedFunc) {
        if (auto clonedParent = loom::utils::getParentModule(clonedFunc))
          insertAfter = clonedParent;
      }
    }

    if (parentModule)
      parentModule.erase();
    else
      originalFunc.erase();
  }

  LogicalResult applyChoices(func::FuncOp func,
                             ArrayRef<BroadcastChoice> choices) {
    auto copyOps = findCopyOpsInFunc(func);
    if (copyOps.size() != choices.size())
      return failure();

    // Apply broadcast choices to each copy operation
    for (size_t i = 0; i < copyOps.size(); ++i) {
      setBroadcastAttribute(copyOps[i].getOperation(), choices[i].values);
    }

    return success();
  }

  ModuleOp module;
  OpBuilder builder;
};

/**
 * @brief Pass to enumerate copy broadcast choices.
 * @details Analyzes loom.copy operations and generates function variants for
 * different broadcast choices. For functions with copy operations, creates
 * clones for each combination of broadcast candidates (including no-broadcast).
 */
struct EnumerateCopyBroadcastPass
    : public PassWrapper<EnumerateCopyBroadcastPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnumerateCopyBroadcastPass)

  EnumerateCopyBroadcastPass() = default;
  EnumerateCopyBroadcastPass(bool analysisOnly) : analysisOnly(analysisOnly) {}

  StringRef getArgument() const override {
    return "loom-enumerate-copy-broadcast";
  }

  StringRef getDescription() const override {
    return "Enumerate broadcast choices for loom.copy operations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    CopyBroadcastEnumerator enumerator(module);
    enumerator.enumerate();
  }

  bool analysisOnly = false;
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createEnumerateCopyBroadcastPass(bool analysisOnly) {
  return std::make_unique<EnumerateCopyBroadcastPass>(analysisOnly);
}
