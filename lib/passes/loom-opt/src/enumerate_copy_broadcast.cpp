/**
 * @file enumerate_copy_broadcast.cpp
 * @brief Implementation for enumerating copy interconnect broadcast choices.
 * @details
 * This pass analyzes loom.copy operations and checks their source operations
 * for spatial reuse information from loom.reinterpret_cast operations.
 * It enumerates all possible interconnect mapping choices (DRAM, horizontal,
 * vertical, all-directions broadcast) and generates function clones for each
 * combination.
 */

#include "Passes.h"
#include "compute_unit_registry.h"
#include "constraint_exporter.h"
#include "constraint_space_utils.h"
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

// Include Loom dialect headers for CopyOp and ReinterpretCastOp
#include "mlir/Interfaces/ViewLikeInterface.h"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

// Include Dataflow dialect headers for SpatialDimOp and InterconnectsOp
#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.h.inc"
#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

using namespace mlir;

namespace {

// Constants for interconnect symbol names (used to identify specific
// interconnect types)
constexpr StringLiteral kHorizontalLinks = "horizontal_links";
constexpr StringLiteral kVerticalLinks = "vertical_links";

/**
 * @brief Extract hardware timing information from the module.
 * @details Walks the module to find df.mat (FPU throughput), df.memory (L1 and
 * DRAM bandwidth), and df.spatial_dim (core counts) operations.
 * @param module The module containing dataflow hardware description.
 * @return HardwareTiming struct with extracted parameters.
 */
static loom::HardwareTiming extractHardwareTiming(ModuleOp module) {
  loom::HardwareTiming hw;

  // Find FPU throughput from df.mat
  module.walk([&](loom::df::MatOp matOp) {
    if (matOp.getName() == "FPU") {
      if (auto throughput = matOp.getThroughput()) {
        hw.fpuThroughput = *throughput;
      }
    }
  });

  // Find L1 and DRAM bandwidth from df.memory
  module.walk([&](loom::df::MemoryOp memOp) {
    StringRef name = memOp.getSymName();
    if (name == "L1") {
      hw.l1Bandwidth = memOp.getBandwidth();
    } else if (name == "DRAM") {
      hw.dramBandwidth = memOp.getBandwidth();
    }
  });

  // Calculate total cores from df.core scaleout
  module.walk([&](loom::df::CoreOp coreOp) {
    int64_t cores = 1;
    for (Value dimValue : coreOp.getScaleout()) {
      if (auto dimOp = dimValue.getDefiningOp<loom::df::SpatialDimOp>()) {
        cores *= dimOp.getSize();
      }
    }
    hw.totalCores = cores;
  });

  return hw;
}

/**
 * @brief Structure representing an interconnect choice for a copy operation.
 * @details Can represent DRAM access, a single interconnect direction, or all
 * directions (broadcast).
 */
struct InterconnectChoice {
  enum Type { DRAM, Single, AllDirections };
  Type type;
  loom::df::InterconnectsOp horizontal; // Used for Single(h) and AllDirections
  loom::df::InterconnectsOp vertical;   // Used for Single(v) and AllDirections

  /**
   * @brief Create a DRAM choice (no interconnect).
   * @return InterconnectChoice representing DRAM access.
   */
  static InterconnectChoice makeDRAM() {
    InterconnectChoice choice;
    choice.type = DRAM;
    choice.horizontal = nullptr;
    choice.vertical = nullptr;
    return choice;
  }

  /**
   * @brief Create a single-direction interconnect choice.
   * @param op The interconnect operation.
   * @return InterconnectChoice representing a single interconnect.
   */
  static InterconnectChoice makeSingle(loom::df::InterconnectsOp op) {
    InterconnectChoice choice;
    choice.type = Single;
    StringRef name = op ? op.getSymName() : StringRef();
    if (name == kHorizontalLinks) {
      choice.horizontal = op;
      choice.vertical = nullptr;
    } else if (name == kVerticalLinks) {
      choice.horizontal = nullptr;
      choice.vertical = op;
    } else {
      choice.horizontal = op;
      choice.vertical = nullptr;
    }
    return choice;
  }

  /**
   * @brief Create an all-directions broadcast choice.
   * @param h The horizontal interconnect operation.
   * @param v The vertical interconnect operation.
   * @return InterconnectChoice representing all-directions broadcast.
   */
  static InterconnectChoice makeAllDirections(loom::df::InterconnectsOp h,
                                              loom::df::InterconnectsOp v) {
    InterconnectChoice choice;
    choice.type = AllDirections;
    choice.horizontal = h;
    choice.vertical = v;
    return choice;
  }

  /**
   * @brief Apply this interconnect choice and update broadcast for a copy op.
   * @param copyOp The copy operation to modify.
   * @param module The module containing df.spatial_dim operations.
   * @param builder OpBuilder for creating attributes.
   * @return true if successfully applied, false otherwise.
   */
  bool apply(loom::CopyToTensorOp copyOp, ModuleOp module,
             OpBuilder &builder) const;
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
 * @brief Get the broadcast attribute from a loom.copy operation.
 * @details Reads the broadcast attribute and converts it to a vector of int64_t
 * values. Supports both DenseI64ArrayAttr and ArrayAttr formats. Defaults to
 * [1, 1] if not found.
 * @param copyOp The copy operation to read the attribute from.
 * @return A vector of broadcast values with at least 2 elements.
 */
static SmallVector<int64_t> getBroadcastAttribute(Operation *copyOp) {
  SmallVector<int64_t> broadcastValues;

  if (auto broadcastAttr =
          copyOp->getAttrOfType<DenseI64ArrayAttr>("broadcast")) {
    broadcastValues.assign(broadcastAttr.asArrayRef().begin(),
                           broadcastAttr.asArrayRef().end());
  } else if (auto broadcastArrayAttr =
                 copyOp->getAttrOfType<ArrayAttr>("broadcast")) {
    for (Attribute elem : broadcastArrayAttr) {
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(elem)) {
        broadcastValues.push_back(intAttr.getInt());
      } else {
        broadcastValues.push_back(1);
      }
    }
  } else {
    broadcastValues = {1, 1};
  }

  // Ensure we have at least 2 elements for 2D broadcast
  while (broadcastValues.size() < 2) {
    broadcastValues.push_back(1);
  }

  return broadcastValues;
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
 * @brief Get the symbol name from an interconnect operation.
 * @param interconnectOp The interconnect operation to get the symbol name from.
 * @return The symbol name as a StringRef, or empty if not found.
 */
static StringRef
getInterconnectSymbolName(loom::df::InterconnectsOp interconnectOp) {
  if (!interconnectOp)
    return StringRef();

  return interconnectOp.getSymName();
}

/**
 * @brief Get a short label for an interconnect operation.
 * @details Returns a single character label based on interconnect type:
 *   - "d" for DRAM (nullptr)
 *   - "h" for horizontal_links
 *   - "v" for vertical_links
 *   - "i" for other interconnects (fallback)
 * @param interconnectOp The interconnect operation (nullptr for DRAM).
 * @return The short label string.
 */
static std::string
getInterconnectShortLabel(loom::df::InterconnectsOp interconnectOp) {
  if (!interconnectOp)
    return "d";
  StringRef name = getInterconnectSymbolName(interconnectOp);
  if (name == kHorizontalLinks)
    return "h";
  if (name == kVerticalLinks)
    return "v";
  return "i";
}

/**
 * @brief Get a short label for an interconnect choice.
 * @details Returns a single character label based on choice type:
 *   - "d" for DRAM
 *   - "a" for AllDirections (both horizontal and vertical)
 *   - "h" for horizontal_links
 *   - "v" for vertical_links
 *   - "i" for other interconnects (fallback)
 * @param choice The interconnect choice.
 * @return The short label string.
 */
static std::string getInterconnectShortLabel(const InterconnectChoice &choice) {
  if (choice.type == InterconnectChoice::DRAM)
    return "d";
  if (choice.type == InterconnectChoice::AllDirections)
    return "a";
  // Single case: check horizontal or vertical
  loom::df::InterconnectsOp op =
      choice.horizontal ? choice.horizontal : choice.vertical;
  return getInterconnectShortLabel(op);
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
 * @brief Find and verify the reinterpret_cast source operation for a copy
 * operation.
 * @details Checks if the copy operation's source operand is defined by a
 * loom.view with spatial_reuse enabled. Returns the
 * view operation if valid.
 * @param copyOp The loom.copy_to_tensor operation to analyze.
 * @return The view operation if found and valid, nullptr otherwise.
 */
static loom::ViewOp findViewSource(loom::CopyToTensorOp copyOp) {
  if (!copyOp)
    return nullptr;

  Value sourceView = copyOp.getSourceView();
  auto viewOp = sourceView.getDefiningOp<loom::ViewOp>();
  if (!viewOp)
    return nullptr;

  if (!viewOp.getSpatialReuse())
    return nullptr;

  return viewOp;
}

/**
 * @brief Find all interconnect operations in the module that match the given
 * spatial dimension.
 * @param module The module containing the interconnect operations.
 * @param spatialDimRef The symbol reference to the spatial dimension to match.
 * @return A vector of matching interconnect operations.
 */
static SmallVector<loom::df::InterconnectsOp>
findMatchingInterconnects(ModuleOp module, SymbolRefAttr spatialDimRef) {
  SmallVector<loom::df::InterconnectsOp> matches;

  module.walk([&](loom::df::InterconnectsOp interconnectOp) {
    auto spatialDimsAttr = interconnectOp.getSpatialDimsAttr();
    if (!spatialDimsAttr)
      return;

    for (Attribute attr : spatialDimsAttr) {
      auto symbolRef = llvm::dyn_cast<SymbolRefAttr>(attr);
      if (symbolRef && symbolRef == spatialDimRef) {
        matches.push_back(interconnectOp);
        return; // Found match, no need to check further
      }
    }
  });

  return matches;
}

/**
 * @brief Find all candidate df.interconnects operations for a loom.copy
 * operation.
 * @details This function checks if the copy operation's source has spatial
 * reuse, and if so, finds all df.interconnects operations that match the
 * spatial dimensions that the reinterpret_cast depends on. Returns
 * InterconnectChoice objects that can represent DRAM, single direction, or
 * all-directions broadcast.
 * @param copyOp The loom.copy operation to analyze.
 * @param module The module containing the df.interconnects operations.
 * @return A vector of candidate InterconnectChoice objects (always includes
 * DRAM option).
 */
static SmallVector<InterconnectChoice>
findCandidateInterconnects(loom::CopyToTensorOp copyOp, ModuleOp module) {
  SmallVector<InterconnectChoice> candidates;
  candidates.push_back(
      InterconnectChoice::makeDRAM()); // Always include DRAM option

  auto viewOp = findViewSource(copyOp);
  if (!viewOp)
    return candidates;

  // Get dynamic offsets from the view operation
  SmallVector<Value> dynamicOffsets(viewOp.getOffsets().begin(),
                                    viewOp.getOffsets().end());
  if (dynamicOffsets.empty())
    return candidates;

  SmallVector<affine::AffineParallelOp> parallelLoops =
      collectEnclosingParallelLoops(copyOp);

  // First collect all raw interconnect operations
  SmallVector<loom::df::InterconnectsOp> rawCandidates;
  for (auto par : parallelLoops) {
    auto mappedToAttr = par->getAttrOfType<SymbolRefAttr>("loom.mapped_to");
    if (!mappedToAttr)
      continue;

    // If offset depends on this loop's IV, different cores need different data,
    // so we cannot use this interconnect for broadcast. Skip it.
    if (checkOffsetDependencyOnIVs(dynamicOffsets, par.getIVs()))
      continue;

    auto matches = findMatchingInterconnects(module, mappedToAttr);
    rawCandidates.append(matches.begin(), matches.end());
  }

  // Now identify horizontal and vertical links
  loom::df::InterconnectsOp horizontalOp = nullptr;
  loom::df::InterconnectsOp verticalOp = nullptr;
  SmallVector<loom::df::InterconnectsOp> otherOps;

  for (auto op : rawCandidates) {
    StringRef name = getInterconnectSymbolName(op);
    if (name == kHorizontalLinks) {
      horizontalOp = op;
    } else if (name == kVerticalLinks) {
      verticalOp = op;
    } else {
      otherOps.push_back(op);
    }
  }

  // If both horizontal and vertical exist, add all-directions option
  if (horizontalOp && verticalOp) {
    candidates.push_back(
        InterconnectChoice::makeAllDirections(horizontalOp, verticalOp));
  }

  // Add individual single-direction choices
  if (horizontalOp) {
    candidates.push_back(InterconnectChoice::makeSingle(horizontalOp));
  }
  if (verticalOp) {
    candidates.push_back(InterconnectChoice::makeSingle(verticalOp));
  }
  for (auto op : otherOps) {
    candidates.push_back(InterconnectChoice::makeSingle(op));
  }

  return candidates;
}

/**
 * @brief Get the size value of a spatial_dim operation by its symbol reference.
 * @details Looks up the df.spatial_dim operation in the module using the symbol
 * reference and returns its size attribute value.
 * @param module The module containing the df.spatial_dim operations.
 * @param symbolRef The symbol reference to the spatial_dim operation.
 * @return The size value if found, or std::nullopt if not found.
 */
static std::optional<uint64_t> getSpatialDimSize(ModuleOp module,
                                                 SymbolRefAttr symbolRef) {
  if (!symbolRef)
    return std::nullopt;

  // Look up the spatial_dim operation by symbol name
  auto spatialDimOp = module.lookupSymbol<loom::df::SpatialDimOp>(symbolRef);
  if (!spatialDimOp)
    return std::nullopt;

  return spatialDimOp.getSize();
}

/**
 * @brief Set the interconnect attribute on a loom.copy operation.
 * @details Creates and sets the interconnect attribute with either the
 * interconnect symbol reference or an empty array for DRAM access.
 * @param copyOp The copy operation to set the attribute on.
 * @param interconnectOp The interconnect operation (nullptr for DRAM).
 * @param builder OpBuilder for creating attributes.
 * @param append If true, appends to existing interconnect array; if false,
 * replaces it.
 * @return true if successfully set, false if interconnectOp is non-null but has
 * no symbol name.
 */
static bool setInterconnectAttribute(loom::CopyToTensorOp copyOp,
                                     loom::df::InterconnectsOp interconnectOp,
                                     OpBuilder &builder, bool append = false) {
  if (interconnectOp) {
    StringRef interconnectName = getInterconnectSymbolName(interconnectOp);
    if (interconnectName.empty())
      return false;

    auto symbolRef = SymbolRefAttr::get(builder.getContext(), interconnectName);

    if (append) {
      // Read existing interconnect array and append
      SmallVector<Attribute> interconnectAttrs;
      if (auto existingAttr =
              copyOp->getAttrOfType<ArrayAttr>("interconnect")) {
        interconnectAttrs.append(existingAttr.begin(), existingAttr.end());
      }
      interconnectAttrs.push_back(symbolRef);
      auto interconnectAttr =
          ArrayAttr::get(builder.getContext(), interconnectAttrs);
      copyOp->setAttr("interconnect", interconnectAttr);
    } else {
      // Replace with new array
      auto interconnectAttr = ArrayAttr::get(builder.getContext(), {symbolRef});
      copyOp->setAttr("interconnect", interconnectAttr);
    }
  } else {
    auto emptyInterconnect = ArrayAttr::get(builder.getContext(), {});
    copyOp->setAttr("interconnect", emptyInterconnect);
  }

  return true;
}

/**
 * @brief Update broadcast attribute based on interconnect type.
 * @details If the interconnect is horizontal_links or vertical_links, updates
 * the corresponding broadcast dimension with the spatial_dim size.
 * @param broadcastValues The broadcast values to update (modified in place).
 * @param interconnectOp The interconnect operation to check.
 * @param module The module containing df.spatial_dim operations.
 */
static void
updateBroadcastForInterconnect(SmallVector<int64_t> &broadcastValues,
                               loom::df::InterconnectsOp interconnectOp,
                               ModuleOp module) {
  if (!interconnectOp)
    return;

  StringRef interconnectName = getInterconnectSymbolName(interconnectOp);
  if (interconnectName != kHorizontalLinks &&
      interconnectName != kVerticalLinks)
    return;

  auto spatialDimsAttr = interconnectOp.getSpatialDimsAttr();
  if (!spatialDimsAttr || spatialDimsAttr.empty())
    return;

  auto firstSpatialDim = llvm::dyn_cast<SymbolRefAttr>(spatialDimsAttr[0]);
  if (!firstSpatialDim)
    return;

  auto sizeOpt = getSpatialDimSize(module, firstSpatialDim);
  if (!sizeOpt.has_value())
    return;

  int64_t size = static_cast<int64_t>(sizeOpt.value());
  if (interconnectName == kHorizontalLinks) {
    broadcastValues[1] = size;
  } else if (interconnectName == kVerticalLinks) {
    broadcastValues[0] = size;
  }
} /**
   * @brief Apply an interconnect and update broadcast attribute to a loom.copy
   * operation.
   */
static bool applyInterconnectAndBroadcastToCopy(
    loom::CopyToTensorOp copyOp, loom::df::InterconnectsOp interconnectOp,
    ModuleOp module, OpBuilder &builder, bool append = false) {
  if (!setInterconnectAttribute(copyOp, interconnectOp, builder, append))
    return false;

  SmallVector<int64_t> broadcastValues = getBroadcastAttribute(copyOp);
  updateBroadcastForInterconnect(broadcastValues, interconnectOp, module);
  setBroadcastAttribute(copyOp, broadcastValues);

  return true;
}

/**
 * @brief Generate a function name based on interconnect choices.
 */
static std::string generateFunctionName(StringRef baseName,
                                        const InterconnectChoice &choice1,
                                        const InterconnectChoice &choice2) {
  std::string label1 = getInterconnectShortLabel(choice1);
  std::string label2 = getInterconnectShortLabel(choice2);
  return baseName.str() + "__" + label1 + "_" + label2;
}

/**
 * @brief Apply this interconnect choice and update broadcast for a copy op.
 */
bool InterconnectChoice::apply(loom::CopyToTensorOp copyOp, ModuleOp module,
                               OpBuilder &builder) const {
  if (type == AllDirections) {
    // First apply horizontal (replaces interconnect attribute)
    if (!applyInterconnectAndBroadcastToCopy(copyOp, horizontal, module,
                                             builder, false))
      return false;
    // Then apply vertical (appends to interconnect attribute)
    return applyInterconnectAndBroadcastToCopy(copyOp, vertical, module,
                                               builder, true);
  }

  // Single or DRAM case
  loom::df::InterconnectsOp op =
      (type == Single) ? (horizontal ? horizontal : vertical) : nullptr;
  return applyInterconnectAndBroadcastToCopy(copyOp, op, module, builder,
                                             false);
}

/**
 * @brief Find all loom.copy operations in a function.
 * @param func The function to search.
 * @return A vector of loom.copy operations found in the function.
 */
static SmallVector<loom::CopyToTensorOp> findCopyOpsInFunc(func::FuncOp func) {
  SmallVector<loom::CopyToTensorOp> copyOps;
  func.walk([&](loom::CopyToTensorOp copyOp) { copyOps.push_back(copyOp); });
  return copyOps;
}

/**
 * @brief Class responsible for enumerating and generating function clones
 * based on copy operation interconnect choices.
 */
class CopyBroadcastEnumerator {
public:
  CopyBroadcastEnumerator(ModuleOp module)
      : module(module), builder(module.getBodyRegion()),
        hwTiming(extractHardwareTiming(module)) {}

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
    if (copyOps.size() != 2)
      return;

    auto candidates1 = findCandidateInterconnects(copyOps[0], module);
    auto candidates2 = findCandidateInterconnects(copyOps[1], module);

    Operation *insertAfter = parentModule;
    for (const auto &choice1 : candidates1) {
      for (const auto &choice2 : candidates2) {
        std::string newName =
            generateFunctionName(originalFunc.getSymName(), choice1, choice2);

        func::FuncOp clonedFunc = loom::utils::cloneFuncWithConstraints(
            builder, originalFunc, newName, moduleAttrs,
            "EnumerateCopyBroadcast",
            [&](func::FuncOp func, loom::ConstraintSpaceOp cs) {
              return applyChoicesAndInjectConstraint(func, cs, choice1,
                                                     choice2);
            },
            insertAfter);

        if (clonedFunc) {
          if (auto clonedParent = loom::utils::getParentModule(clonedFunc))
            insertAfter = clonedParent;
        }
      }
    }

    if (parentModule)
      parentModule.erase();
    else
      originalFunc.erase();
  }

  LogicalResult applyChoicesAndInjectConstraint(func::FuncOp func,
                                                loom::ConstraintSpaceOp csOp,
                                                const InterconnectChoice &c1,
                                                const InterconnectChoice &c2) {
    auto copyOps = findCopyOpsInFunc(func);
    if (copyOps.size() != 2)
      return failure();

    // Apply broadcast choices
    OpBuilder b1(copyOps[0].getOperation());
    if (!c1.apply(copyOps[0], module, b1))
      return failure();

    OpBuilder b2(copyOps[1].getOperation());
    if (!c2.apply(copyOps[1], module, b2))
      return failure();

    // Inject compute-memory constraint if we have valid hardware timing
    if (csOp && hwTiming.fpuThroughput > 0 && hwTiming.totalCores > 0) {
      // Get broadcast values for each copy operation after choices are applied
      auto broadcastA = getBroadcastAttribute(copyOps[0]);
      auto broadcastB = getBroadcastAttribute(copyOps[1]);

      // Calculate effective bandwidths based on broadcast patterns
      int64_t bwA =
          hwTiming.getEffectiveBandwidth(broadcastA[0], broadcastA[1]);
      int64_t bwB =
          hwTiming.getEffectiveBandwidth(broadcastB[0], broadcastB[1]);

      // Add compute-memory constraint: BW_B*sizeA + BW_A*sizeB -
      // BW_A*BW_B*compute/T <= 0 Variable order: BM (index 0), BN (index 1), BK
      // (index 2)
      loom::lcs::addComputeMemoryConstraint(csOp, {"BM", "BN", "BK"}, bwA, bwB,
                                            hwTiming.fpuThroughput,
                                            /*elemSize=*/4, /*flopCoeff=*/2);
    }

    return success();
  }

  ModuleOp module;
  OpBuilder builder;
  loom::HardwareTiming hwTiming;
};

/**
 * @brief Pass to enumerate copy interconnect broadcast choices.
 * @details Analyzes loom.copy operations and generates function variants for
 * different interconnect broadcast choices. For functions with exactly two copy
 * operations, creates clones for each combination of interconnect candidates
 * (including DRAM option).
 */
struct EnumerateCopyBroadcastPass
    : public PassWrapper<EnumerateCopyBroadcastPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnumerateCopyBroadcastPass)

  EnumerateCopyBroadcastPass() = default;
  EnumerateCopyBroadcastPass(bool analysisOnly) : analysisOnly(analysisOnly) {}

  /**
   * @brief Get the command-line argument name for this pass.
   * @return The argument name string.
   */
  StringRef getArgument() const override {
    return "loom-enumerate-copy-broadcast";
  }

  /**
   * @brief Get the description of this pass.
   * @return The description string.
   */
  StringRef getDescription() const override {
    return "Enumerate interconnect broadcast choices for loom.copy operations";
  }

  /**
   * @brief Execute the pass over the module.
   * @details For each function with exactly two loom.copy operations, finds
   * candidate interconnects for each copy and generates function clones for all
   * combinations (Cartesian product of candidates1 × candidates2).
   *
   * Note: This pass operates directly on the existing module without creating
   * a new one, so module-level attributes are automatically preserved.
   */
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Perform enumeration and cloning
    CopyBroadcastEnumerator enumerator(module);
    enumerator.enumerate();

    // Export simplified constraints to JSON for each constraint space
    llvm::SmallVector<std::string, 8> allJsonStrings;
    module.walk([&](ModuleOp wrapperModule) {
      if (auto csOp = loom::lcs::findConstraintSpace(wrapperModule)) {
        std::string passName = "EnumerateCopyBroadcast";
        auto passNameAttr =
            wrapperModule->getAttrOfType<StringAttr>("loom.pass_name");
        if (passNameAttr)
          passName = passNameAttr.getValue().str();

        allJsonStrings.push_back(
            loom::lcs::exportConstraintSpaceToJson(csOp, passName));
      }
    });

    if (!allJsonStrings.empty()) {
      llvm::errs() << "[\n";
      for (size_t i = 0; i < allJsonStrings.size(); ++i) {
        // Indent the internal JSON
        std::string indented;
        llvm::raw_string_ostream os(indented);
        bool firstLine = true;
        for (char c : allJsonStrings[i]) {
          if (firstLine) {
            os << "  ";
            firstLine = false;
          }
          os << c;
          if (c == '\n')
            os << "  ";
        }
        llvm::errs() << os.str();
        if (i < allJsonStrings.size() - 1)
          llvm::errs() << ",";
        llvm::errs() << "\n";
      }
      llvm::errs() << "]\n";
    }
  }

  /// Flag indicating whether this pass should only perform analysis (currently
  /// unused).
  bool analysisOnly = false;
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createEnumerateCopyBroadcastPass(bool analysisOnly) {
  return std::make_unique<EnumerateCopyBroadcastPass>(analysisOnly);
}
