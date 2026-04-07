/**
 * @file enumerate_copy_broadcast.cpp
 * @brief Implementation for enumerating copy broadcast choices.
 * @details
 * This pass analyzes loom.copy operations and checks their source operations
 * for spatial reuse information from loom.subview operations.
 * It enumerates all possible broadcast choices based on per-dimension,
 * multi-level analysis and generates function clones for each combination.
 *
 * For each physical dimension, the broadcast coefficient is the product of
 * upper bounds of all contiguous independent levels starting from level 0.
 * If level 0 is dependent, no broadcast on that dim. If levels 0..K are
 * independent but level K+1 is dependent, the coefficient = UB0 * ... * UBK.
 *
 * Hardware dimension info is read from adl.arch.scale and adl.spatial_dim
 * operations in the outer module.
 */

#include "Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

// Include Loom dialect headers for CopyOp and SubviewOp
#include "mlir/Interfaces/ViewLikeInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

// Include ADL dialect headers for adl.spatial_dim, adl.arch.scale
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
  SmallVector<int64_t> values; // e.g., {1, 1}, {1, 2}, {8, 1}, {8, 8}
  std::string label;           // "n", "dim_y_level0_bc2", etc.
};

/**
 * @brief Info about a spatial parallel loop with its dim/level metadata.
 */
struct SpatialLoopEntry {
  affine::AffineParallelOp parOp;
  std::string physicalDim; // e.g., "dim_x", "dim_y"
  int64_t logicalLevel;    // 0, 1, 2, ...
  int64_t upperBound;      // constant UB of this loop
};

/**
 * @brief Per-dimension broadcast analysis result.
 */
struct DimBroadcastResult {
  std::string dimName;
  int64_t coefficient;  // product of UBs of contiguous independent levels from 0
  int64_t highestLevel; // highest contiguous independent level (-1 if none)
  bool canBroadcast;    // coefficient > 1
};

/**
 * @brief Check whether a value depends (transitively) on a target value.
 * @note This function is duplicated in evaluate_reuse.cpp. Consolidate into a 
 * shared utility header in a future refactor.
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
 * @brief Check if any offset value depends on any of the given induction
 * variables.
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
 * @brief Extract the constant upper bound from a spatial parallel loop.
 * @details Spatial parallel loops always have constant upper bounds
 * (hardware tile sizes). Asserts if the bound is not constant.
 */
static int64_t getConstantUpperBound(affine::AffineParallelOp par) {
  AffineMap ubMap = par.getUpperBoundsMap();
  assert(ubMap.getNumResults() == 1 && "spatial loop must have single UB");
  auto constExpr = dyn_cast<AffineConstantExpr>(ubMap.getResult(0));
  assert(constExpr && "spatial loop must have constant UB");
  return constExpr.getValue();
}

/**
 * @brief Collect ordered mesh dimension names from adl.arch.scale.
 * @details The spatial dims used in adl.arch.scale define the mesh dimensions
 * and their ordering (e.g., ["dim_x", "dim_y"]).
 */
static SmallVector<std::string> collectMeshDimNames(ModuleOp outerModule) {
  SmallVector<std::string> dimNames;
  outerModule.walk([&](adl::ArchScaleOp scaleOp) {
    for (Value operand : scaleOp.getSpatialDims()) {
      if (auto dimOp = operand.getDefiningOp<adl::SpatialDimOp>())
        dimNames.push_back(dimOp.getSymName().str());
    }
  });
  return dimNames;
}

/**
 * @brief Collect enclosing spatial parallel loops grouped by physical dim.
 * @details Walks up the parent chain from an operation, collecting all
 * affine.parallel loops with loom.physical_dim and loom.logical_level
 * attributes. Results are grouped by dim name and sorted by level ascending.
 */
static llvm::StringMap<SmallVector<SpatialLoopEntry>>
collectSpatialLoopsByDim(Operation *op) {
  llvm::StringMap<SmallVector<SpatialLoopEntry>> dimLoops;

  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto par = dyn_cast<affine::AffineParallelOp>(parent);
    if (!par)
      continue;

    auto dimAttr = par->getAttrOfType<SymbolRefAttr>("loom.physical_dim");
    auto levelAttr = par->getAttrOfType<IntegerAttr>("loom.logical_level");
    if (!dimAttr || !levelAttr)
      continue;

    int64_t ub = getConstantUpperBound(par);
    std::string dimName = dimAttr.getRootReference().getValue().str();
    int64_t level = levelAttr.getInt();

    dimLoops[dimName].push_back({par, dimName, level, ub});
  }

  // Sort each dim's entries by level ascending
  for (auto &entry : dimLoops) {
    llvm::sort(entry.second, [](const SpatialLoopEntry &a,
                                const SpatialLoopEntry &b) {
      return a.logicalLevel < b.logicalLevel;
    });
  }

  return dimLoops;
}

/**
 * @brief Compute per-dim broadcast coefficient using contiguous level
 * independence.
 * @details Starting from level 0, accumulates UBs of consecutive independent
 * levels. Stops at the first dependent level (chain-breaking rule).
 */
static DimBroadcastResult
computeDimBroadcastCoeff(StringRef dimName,
                         SmallVectorImpl<SpatialLoopEntry> &levelEntries,
                         ArrayRef<Value> offsets) {
  int64_t coeff = 1;
  int64_t highestLevel = -1;

  for (auto &entry : levelEntries) {
    bool independent = !checkOffsetDependencyOnIVs(offsets, entry.parOp.getIVs());
    if (!independent)
      break; // chain broken - stop accumulating
    coeff *= entry.upperBound;
    highestLevel = entry.logicalLevel;
  }

  return {dimName.str(), coeff, highestLevel, coeff > 1};
}

/**
 * @brief Find and verify the subview operand for a copy operation.
 * @details Every loom.copy must have exactly one operand (source or
 * destination) produced by a loom.subview. Returns the subview only if it has
 * spatial_reuse enabled (broadcast candidate). Returns nullptr otherwise.
 */
static loom::SubviewOp findSubviewSource(loom::CopyOp copyOp) {
  auto srcSubview = copyOp.getSource().getDefiningOp<loom::SubviewOp>();
  auto dstSubview = copyOp.getDestination().getDefiningOp<loom::SubviewOp>();
  assert((srcSubview || dstSubview) &&
         "loom.copy must have a loom.subview operand");

  // For broadcast analysis, prefer the source subview (load direction)
  loom::SubviewOp subviewOp = srcSubview ? srcSubview : dstSubview;

  if (!subviewOp.getSpatialReuse())
    return nullptr;

  return subviewOp;
}

/**
 * @brief Find all broadcast candidate choices for a loom.copy operation.
 * @details Analyzes the copy's subview for spatial reuse, then for each mesh
 * dimension checks which contiguous levels (from level 0 up) have IVs that the
 * subview offsets do NOT depend on. The broadcast coefficient per dim is the
 * product of UBs of those contiguous independent levels.
 *
 * Generates all 2^N subsets of broadcastable dimensions as choices (plus the
 * no-broadcast choice).
 */
static SmallVector<BroadcastChoice>
findCopyBroadcastCandidates(loom::CopyOp copyOp, ModuleOp /*outerModule*/,
                            ArrayRef<std::string> meshDimNames) {
  size_t numDims = meshDimNames.size();
  SmallVector<BroadcastChoice> candidates;

  // Always include no-broadcast option
  candidates.push_back({SmallVector<int64_t>(numDims, 1), "n"});

  auto subviewOp = findSubviewSource(copyOp);
  if (!subviewOp)
    return candidates;

  SmallVector<Value> offsets(subviewOp.getOffsets().begin(),
                             subviewOp.getOffsets().end());
  if (offsets.empty())
    return candidates;

  // Collect spatial loops grouped by physical dim
  auto dimLoops = collectSpatialLoopsByDim(copyOp);

  // Compute per-dim broadcast results in mesh dim order
  SmallVector<DimBroadcastResult> dimResults;
  for (const auto &dimName : meshDimNames) {
    auto it = dimLoops.find(dimName);
    if (it != dimLoops.end())
      dimResults.push_back(
          computeDimBroadcastCoeff(dimName, it->second, offsets));
    else
      dimResults.push_back({dimName, 1, -1, false});
  }

  // Collect broadcastable dim indices
  SmallVector<size_t> bcDims;
  for (size_t i = 0; i < dimResults.size(); ++i) {
    if (dimResults[i].canBroadcast)
      bcDims.push_back(i);
  }

  // Enumerate all non-empty subsets of broadcastable dims
  size_t numBroadcastable = bcDims.size();
  for (size_t mask = 1; mask < (1u << numBroadcastable); ++mask) {
    SmallVector<int64_t> values(numDims, 1);
    std::string label;

    for (size_t bit = 0; bit < numBroadcastable; ++bit) {
      if (mask & (1u << bit)) {
        size_t idx = bcDims[bit];
        values[idx] = dimResults[idx].coefficient;
        if (!label.empty())
          label += "_";
        label += dimResults[idx].dimName + "_level" +
                 std::to_string(dimResults[idx].highestLevel) + "_bc" +
                 std::to_string(dimResults[idx].coefficient);
      }
    }

    candidates.push_back({values, label});
  }

  return candidates;
}

/**
 * @brief Generate a function name based on broadcast choices.
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
      : module(module), builder(module.getBodyRegion()),
        meshDimNames(collectMeshDimNames(module)) {}

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
      allCandidates.push_back(
          findCopyBroadcastCandidates(copyOp, module, meshDimNames));
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
  SmallVector<std::string> meshDimNames;
};

/**
 * @brief Pass to enumerate copy broadcast choices.
 */
struct EnumerateCopyBroadcastPass
    : public PassWrapper<EnumerateCopyBroadcastPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnumerateCopyBroadcastPass)

  EnumerateCopyBroadcastPass() = default;

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

};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createEnumerateCopyBroadcastPass() {
  return std::make_unique<EnumerateCopyBroadcastPass>();
}
