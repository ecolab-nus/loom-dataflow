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
#include "hw_dim_splitter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "ssa_utils.h"

// Include Loom dialect headers for CopyOp and SubviewOp
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "LoomInterfaces.h.inc"
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
  SmallVector<int64_t> values; // broadcast factor per dim, e.g., {1, 1}, {8, 1}
  // For each physical dim, array-indices into axis.ivs/tileSizes to override.
  // Empty entry = no broadcast on that dim.
  SmallVector<SmallVector<unsigned>> overrideLevelIndices;
  std::string label; // "n", "dim_y_level0_bc2", etc.
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
  SmallVector<unsigned> independentLevelIndices; // 0-based array indices into axis.ivs
};


/**
 * @brief Check if any offset value depends on any of the given induction
 * variables.
 */
static bool checkOffsetDependencyOnIVs(ArrayRef<Value> offsets,
                                       ValueRange ivs) {
  for (Value iv : ivs) {
    for (Value offset : offsets) {
      if (loom::utils::dependsOn(offset, iv)) {
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
  SmallVector<unsigned> independentLevelIndices;

  for (unsigned idx = 0; idx < levelEntries.size(); ++idx) {
    auto &entry = levelEntries[idx];
    bool independent = !checkOffsetDependencyOnIVs(offsets, entry.parOp.getIVs());
    if (!independent)
      break; // chain broken - stop accumulating
    coeff *= entry.upperBound;
    highestLevel = entry.logicalLevel;
    independentLevelIndices.push_back(idx);
  }

  return {dimName.str(), coeff, highestLevel, coeff > 1,
          std::move(independentLevelIndices)};
}

/**
 * @brief Find and verify the source subview for a copy operation.
 * @details Broadcast analysis is only applicable when the copy source is a
 * loom.subview (typically DRAM->L1 load). Destination-only subview cases
 * (e.g., L1->DRAM write-back) are intentionally excluded.
 */
static loom::SubviewOp findSourceSubviewForBroadcast(loom::CopyOp copyOp) {
  auto srcSubview = copyOp.getSource().getDefiningOp<loom::SubviewOp>();
  if (!srcSubview)
    return nullptr;

  if (!srcSubview.getSpatialReuse())
    return nullptr;

  return srcSubview;
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
  candidates.push_back(
      {SmallVector<int64_t>(numDims, 1),
       SmallVector<SmallVector<unsigned>>(numDims), "n"});

  // Skip broadcast enumeration for copies that write out gather+reduce results.
  // The chain is: copy-src = bufferize_to_memref(linalg.generic(ins=[gather_result]))
  auto isProducedByGather = [](Value v) {
    if (!v) return false;
    Operation *op = v.getDefiningOp();
    if (!op) return false;
    if (isa<loom::GatherOp>(op)) return true;
    if (auto b = dyn_cast<loom::BufferizeToMemrefOp>(op)) {
      Operation *srcOp = b.getSource().getDefiningOp();
      if (!srcOp) return false;
      if (isa<loom::GatherOp>(srcOp)) return true;
      // One more level: through linalg.generic consuming a gather result
      for (Value operand : srcOp->getOperands())
        if (operand.getDefiningOp() && isa<loom::GatherOp>(operand.getDefiningOp()))
          return true;
    }
    return false;
  };

  if (isProducedByGather(copyOp.getSource()) ||
      isProducedByGather(copyOp.getDestination())) {
    return candidates;
  }

  auto subviewOp = findSourceSubviewForBroadcast(copyOp);
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
      dimResults.push_back({dimName, 1, -1, false, {}});
  }

  // Build the single best broadcast choice: enable all broadcastable dims.
  // This avoids combinatorial explosion from enumerating subsets.
  SmallVector<int64_t> bestValues(numDims, 1);
  SmallVector<SmallVector<unsigned>> bestOverrides(numDims);
  std::string bestLabel;

  for (size_t i = 0; i < dimResults.size(); ++i) {
    if (!dimResults[i].canBroadcast)
      continue;
    bestValues[i] = dimResults[i].coefficient;
    bestOverrides[i] = dimResults[i].independentLevelIndices;
    if (!bestLabel.empty())
      bestLabel += "_";
    bestLabel += dimResults[i].dimName + "_level" +
                 std::to_string(dimResults[i].highestLevel) + "_bc" +
                 std::to_string(dimResults[i].coefficient);
  }

  if (!bestLabel.empty()) {
    // At least one dim is broadcastable: return the best choice only.
    candidates.clear();
    candidates.push_back({bestValues, std::move(bestOverrides), bestLabel});
  }
  // Otherwise candidates already holds the single no-broadcast option.

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

      func::FuncOp clonedFunc = loom::utils::cloneFunc(
          builder, originalFunc, newName, moduleAttrs,
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

    for (size_t i = 0; i < copyOps.size(); ++i) {
      loom::CopyOp copyOp = copyOps[i];
      const BroadcastChoice &choice = choices[i];

      // Reconstruct mesh coordinate system from enclosing loop attributes.
      auto meshCoords = loom::MeshCoordinateSystem::fromEnclosingLoops(
          copyOp.getOperation(), meshDimNames);

      OpBuilder builder(copyOp.getOperation());
      Location loc = copyOp.getLoc();

      // Emit UL/LR values for one axis given the override level indices.
      auto computeAxisBounds =
          [&](const loom::AxisLinearIndex &axis,
              unsigned dimIdx) -> std::pair<Value, Value> {
        if (axis.ivs.empty()) {
          Value zero =
              arith::ConstantIndexOp::create(builder, loc, 0).getResult();
          return {zero, zero};
        }
        const SmallVector<unsigned> &overrideLvls =
            (dimIdx < choice.overrideLevelIndices.size())
                ? choice.overrideLevelIndices[dimIdx]
                : SmallVector<unsigned>{};
        if (overrideLvls.empty()) {
          // No broadcast: UL == LR == current mesh position.
          Value pos = meshCoords.emitLinearIndex(builder, loc, axis);
          return {pos, pos};
        }
        llvm::DenseMap<unsigned, int64_t> ulOvr, lrOvr;
        for (unsigned lvlIdx : overrideLvls) {
          ulOvr[lvlIdx] = 0;
          lrOvr[lvlIdx] = axis.tileSizes[lvlIdx] - 1;
        }
        Value ul = meshCoords.emitLinearIndexWithMultiOverride(
            builder, loc, axis, ulOvr);
        Value lr = meshCoords.emitLinearIndexWithMultiOverride(
            builder, loc, axis, lrOvr);
        return {ul, lr};
      };

      auto [ul_x, lr_x] = computeAxisBounds(meshCoords.xAxis, 0);
      auto [ul_y, lr_y] = computeAxisBounds(meshCoords.yAxis, 1);

      // Create replacement CopyOp with static area and UL/LR bounds.
      auto newAreaAttr = builder.getDenseI64ArrayAttr(choice.values);
      loom::CopyOp::create(builder, loc, copyOp.getSource(),
                           copyOp.getDestination(),
                           ValueRange{},
                           copyOp.getSrcMemSpaceAttr(),
                           copyOp.getDstMemSpaceAttr(), newAreaAttr,
                           ul_x, ul_y, lr_x, lr_y);
      copyOp.erase();
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
