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

#include "enumerate_copy_broadcast.h"
#include "utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

// Include Loom dialect headers for CopyOp and ReinterpretCastOp
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

// Include Dataflow dialect headers for SpatialDimOp and InterconnectsOp
#define GET_TYPEDEF_CLASSES
#include "DataflowTypes.h.inc"
#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

using namespace mlir;

namespace {

// Constants for interconnect symbol names (used to identify specific interconnect types)
constexpr StringLiteral kHorizontalLinks = "horizontal_links";
constexpr StringLiteral kVerticalLinks = "vertical_links";

/**
 * @brief Structure representing an interconnect choice for a copy operation.
 * @details Can represent DRAM access, a single interconnect direction, or all directions (broadcast).
 */
struct InterconnectChoice {
  enum Type { DRAM, Single, AllDirections };
  Type type;
  loom::df::InterconnectsOp horizontal;  // Used for Single(h) and AllDirections
  loom::df::InterconnectsOp vertical;    // Used for Single(v) and AllDirections
  
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
 * @details Reads the broadcast attribute and converts it to a vector of int64_t values.
 * Supports both DenseI64ArrayAttr and ArrayAttr formats. Defaults to [1, 1] if not found.
 * @param copyOp The copy operation to read the attribute from.
 * @return A vector of broadcast values with at least 2 elements.
 */
static SmallVector<int64_t> getBroadcastAttribute(Operation *copyOp) {
  SmallVector<int64_t> broadcastValues;
  
  if (auto broadcastAttr = copyOp->getAttrOfType<DenseI64ArrayAttr>("broadcast")) {
    broadcastValues.assign(broadcastAttr.asArrayRef().begin(), 
                           broadcastAttr.asArrayRef().end());
  } else if (auto broadcastArrayAttr = copyOp->getAttrOfType<ArrayAttr>("broadcast")) {
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
 * @details Creates an ArrayAttr of IntegerAttr from the broadcast values and sets it on the operation.
 * @param copyOp The copy operation to set the attribute on.
 * @param broadcastValues The broadcast values to set.
 */
static void setBroadcastAttribute(Operation *copyOp, ArrayRef<int64_t> broadcastValues) {
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
static StringRef getInterconnectSymbolName(loom::df::InterconnectsOp interconnectOp) {
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
static std::string getInterconnectShortLabel(loom::df::InterconnectsOp interconnectOp) {
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
  loom::df::InterconnectsOp op = choice.horizontal ? choice.horizontal : choice.vertical;
  return getInterconnectShortLabel(op);
}

/**
 * @brief Collect all enclosing affine.parallel loops for an operation.
 * @details Walks up the parent chain to find all affine.parallel operations that enclose the given operation.
 * @param op The operation to find enclosing loops for.
 * @return A vector of enclosing affine.parallel operations in order from outermost to innermost.
 */
static SmallVector<affine::AffineParallelOp> collectEnclosingParallelLoops(Operation *op) {
  SmallVector<affine::AffineParallelOp> parallelLoops;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto par = dyn_cast<affine::AffineParallelOp>(parent))
      parallelLoops.push_back(par);
  }
  return parallelLoops;
}

/**
 * @brief Check if any offset value depends on any of the given induction variables.
 * @param offsets The offset values to check.
 * @param ivs The induction variables to check dependency against.
 * @return true if any offset depends on any IV, false otherwise.
 */
static bool checkOffsetDependencyOnIVs(ArrayRef<Value> offsets, ValueRange ivs) {
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
 * @brief Find and verify the reinterpret_cast source operation for a copy operation.
 * @details Checks if the copy operation's source operand is defined by a loom.reinterpret_cast
 * with spatial_reuse enabled. Returns the reinterpret_cast operation if valid.
 * @param copyOp The loom.copy operation to analyze.
 * @return The reinterpret_cast operation if found and valid, nullptr otherwise.
 */
static loom::ReinterpretCastOp findReinterpretCastSource(loom::CopyOp copyOp) {
  if (!copyOp)
    return nullptr;
  
  Value src = copyOp.getSrc();
  auto reinterpretCastOp = src.getDefiningOp<loom::ReinterpretCastOp>();
  if (!reinterpretCastOp)
    return nullptr;
  
  if (!reinterpretCastOp.getSpatialReuse())
    return nullptr;
  
  return reinterpretCastOp;
}

/**
 * @brief Find all interconnect operations in the module that match the given spatial dimension.
 * @param module The module containing the interconnect operations.
 * @param spatialDimRef The symbol reference to the spatial dimension to match.
 * @return A vector of matching interconnect operations.
 */
static SmallVector<loom::df::InterconnectsOp> findMatchingInterconnects(ModuleOp module, 
                                                                         SymbolRefAttr spatialDimRef) {
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
 * @brief Find all candidate df.interconnects operations for a loom.copy operation.
 * @details This function checks if the copy operation's source has spatial reuse,
 * and if so, finds all df.interconnects operations that match the spatial dimensions
 * that the reinterpret_cast depends on. Returns InterconnectChoice objects that can
 * represent DRAM, single direction, or all-directions broadcast.
 * @param copyOp The loom.copy operation to analyze.
 * @param module The module containing the df.interconnects operations.
 * @return A vector of candidate InterconnectChoice objects (always includes DRAM option).
 */
static SmallVector<InterconnectChoice> findCandidateInterconnects(loom::CopyOp copyOp,
                                                                   ModuleOp module) {
  SmallVector<InterconnectChoice> candidates;
  candidates.push_back(InterconnectChoice::makeDRAM()); // Always include DRAM option

  auto reinterpretCastOp = findReinterpretCastSource(copyOp);
  if (!reinterpretCastOp)
    return candidates;
  
  // Get dynamic offsets from the reinterpret_cast operation
  SmallVector<Value> dynamicOffsets(reinterpretCastOp.getOffsets().begin(),
                                     reinterpretCastOp.getOffsets().end());
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
    candidates.push_back(InterconnectChoice::makeAllDirections(horizontalOp, verticalOp));
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
 * @details Looks up the df.spatial_dim operation in the module using the symbol reference
 * and returns its size attribute value.
 * @param module The module containing the df.spatial_dim operations.
 * @param symbolRef The symbol reference to the spatial_dim operation.
 * @return The size value if found, or std::nullopt if not found.
 */
static std::optional<uint64_t> getSpatialDimSize(ModuleOp module, SymbolRefAttr symbolRef) {
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
 * @details Creates and sets the interconnect attribute with either the interconnect symbol
 * reference or an empty array for DRAM access.
 * @param copyOp The copy operation to set the attribute on.
 * @param interconnectOp The interconnect operation (nullptr for DRAM).
 * @param builder OpBuilder for creating attributes.
 * @param append If true, appends to existing interconnect array; if false, replaces it.
 * @return true if successfully set, false if interconnectOp is non-null but has no symbol name.
 */
static bool setInterconnectAttribute(loom::CopyOp copyOp, 
                                     loom::df::InterconnectsOp interconnectOp,
                                     OpBuilder &builder,
                                     bool append = false) {
  if (interconnectOp) {
    StringRef interconnectName = getInterconnectSymbolName(interconnectOp);
    if (interconnectName.empty())
      return false;
    
    auto symbolRef = SymbolRefAttr::get(builder.getContext(), interconnectName);
    
    if (append) {
      // Read existing interconnect array and append
      SmallVector<Attribute> interconnectAttrs;
      if (auto existingAttr = copyOp->getAttrOfType<ArrayAttr>("interconnect")) {
        interconnectAttrs.append(existingAttr.begin(), existingAttr.end());
      }
      interconnectAttrs.push_back(symbolRef);
      auto interconnectAttr = ArrayAttr::get(builder.getContext(), interconnectAttrs);
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
 * @details If the interconnect is horizontal_links or vertical_links, updates the corresponding
 * broadcast dimension with the spatial_dim size.
 * @param broadcastValues The broadcast values to update (modified in place).
 * @param interconnectOp The interconnect operation to check.
 * @param module The module containing df.spatial_dim operations.
 */
static void updateBroadcastForInterconnect(SmallVector<int64_t> &broadcastValues,
                                           loom::df::InterconnectsOp interconnectOp,
                                           ModuleOp module) {
  if (!interconnectOp)
    return;
  
  StringRef interconnectName = getInterconnectSymbolName(interconnectOp);
  if (interconnectName != kHorizontalLinks && interconnectName != kVerticalLinks)
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
}

/**
 * @brief Apply an interconnect and update broadcast attribute to a loom.copy operation.
 * @details Sets the interconnect attribute and updates the broadcast attribute if needed.
 * If interconnectOp is nullptr (indicating DRAM access), creates an empty array attribute.
 * 
 * For horizontal_links or vertical_links interconnects, updates the corresponding
 * broadcast dimension with the spatial_dim size:
 * - horizontal_links -> updates broadcast[1]
 * - vertical_links -> updates broadcast[0]
 * 
 * @param copyOp The loom.copy operation to modify.
 * @param interconnectOp The interconnect operation to apply (can be nullptr for DRAM).
 * @param module The module containing df.spatial_dim operations.
 * @param builder OpBuilder for creating new operations.
 * @param append If true, appends interconnect to existing array; if false, replaces it.
 * @return true if successfully applied, false otherwise.
 */
static bool applyInterconnectAndBroadcastToCopy(loom::CopyOp copyOp, 
                                                loom::df::InterconnectsOp interconnectOp,
                                                ModuleOp module, OpBuilder &builder,
                                                bool append = false) {
  if (!setInterconnectAttribute(copyOp, interconnectOp, builder, append))
    return false;
  
  SmallVector<int64_t> broadcastValues = getBroadcastAttribute(copyOp);
  updateBroadcastForInterconnect(broadcastValues, interconnectOp, module);
  setBroadcastAttribute(copyOp, broadcastValues);
  
  return true;
}

/**
 * @brief Apply all-directions broadcast to a loom.copy operation.
 * @details Applies both horizontal and vertical interconnects by calling
 * applyInterconnectAndBroadcastToCopy twice. First applies horizontal,
 * then appends vertical to the interconnect attribute array.
 * 
 * @param copyOp The loom.copy operation to modify.
 * @param horizontalOp The horizontal interconnect operation.
 * @param verticalOp The vertical interconnect operation.
 * @param module The module containing df.spatial_dim operations.
 * @param builder OpBuilder for creating new operations.
 * @return true if successfully applied both directions, false otherwise.
 */
static bool applyAllDirectionsInterconnectToCopy(loom::CopyOp copyOp,
                                                 loom::df::InterconnectsOp horizontalOp,
                                                 loom::df::InterconnectsOp verticalOp,
                                                 ModuleOp module, OpBuilder &builder) {
  // First call: apply horizontal (replaces interconnect attribute)
  if (!applyInterconnectAndBroadcastToCopy(copyOp, horizontalOp, module, builder, false))
    return false;
  
  // Second call: apply vertical (appends to interconnect attribute)
  return applyInterconnectAndBroadcastToCopy(copyOp, verticalOp, module, builder, true);
}

/**
 * @brief Find all loom.copy operations in a function.
 * @param func The function to search.
 * @return A vector of loom.copy operations found in the function.
 */
static SmallVector<loom::CopyOp> findCopyOpsInFunc(func::FuncOp func) {
  SmallVector<loom::CopyOp> copyOps;
  func.walk([&](loom::CopyOp copyOp) {
    copyOps.push_back(copyOp);
  });
  return copyOps;
}

/**
 * @brief Generate a function name based on interconnect choices.
 * @details Creates a new function name by appending short interconnect labels to the base name.
 * Uses labels: "d" for DRAM, "h" for horizontal, "v" for vertical, "a" for all directions.
 * @param baseName The base function name.
 * @param choice1 The first interconnect choice.
 * @param choice2 The second interconnect choice.
 * @return The new function name with interconnect labels appended.
 */
static std::string generateFunctionName(StringRef baseName,
                                        const InterconnectChoice &choice1,
                                        const InterconnectChoice &choice2) {
  std::string label1 = getInterconnectShortLabel(choice1);
  std::string label2 = getInterconnectShortLabel(choice2);
  return baseName.str() + "__" + label1 + "_" + label2;
}

/**
 * @brief Pass to enumerate copy interconnect broadcast choices.
 * @details Analyzes loom.copy operations and generates function variants for different
 * interconnect broadcast choices. For functions with exactly two copy operations, creates
 * clones for each combination of interconnect candidates (including DRAM option).
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
   * @details For each function with exactly two loom.copy operations, finds candidate
   * interconnects for each copy and generates function clones for all combinations
   * (Cartesian product of candidates1 × candidates2).
   * 
   * Note: This pass operates directly on the existing module without creating
   * a new one, so module-level attributes are automatically preserved.
   */
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder moduleBuilder(module.getBodyRegion());

    SmallVector<func::FuncOp> funcs = loom::utils::collectFunctions(module);

    for (func::FuncOp originalFunc : funcs) {
      // Get the attributes from the parent module of the function
      ModuleOp parentModule = loom::utils::getParentModule(originalFunc);
      DictionaryAttr moduleAttrs = nullptr;
      if (parentModule) {
        moduleAttrs = parentModule->getAttrDictionary();
      }
      
      auto copyOps = findCopyOpsInFunc(originalFunc);
      if (copyOps.size() != 2)
        continue;

      loom::CopyOp copyOp1 = copyOps[0];
      loom::CopyOp copyOp2 = copyOps[1];

      auto candidates1 = findCandidateInterconnects(copyOp1, module);
      auto candidates2 = findCandidateInterconnects(copyOp2, module);

      // Track insertion at module level
      Operation *insertAfter = parentModule;
      for (const InterconnectChoice &choice1 : candidates1) {
        for (const InterconnectChoice &choice2 : candidates2) {
          std::string newName = generateFunctionName(originalFunc.getSymName(),
                                                      choice1, choice2);
          
          // Clone and modify the function with interconnect choices and module wrapper
          func::FuncOp clonedFunc = loom::utils::cloneModifyAndInsertFunctionWithModuleWrapper(
              moduleBuilder, originalFunc, newName, moduleAttrs,
              [&](func::FuncOp func) -> LogicalResult {
                // Find copy operations in the cloned function
                auto clonedCopyOps = findCopyOpsInFunc(func);
                if (clonedCopyOps.size() != 2) {
                  return failure();
                }
                
                loom::CopyOp clonedCopyOp1 = clonedCopyOps[0];
                OpBuilder builder(clonedCopyOp1);
                
                // Apply first interconnect choice
                bool success1 = false;
                if (choice1.type == InterconnectChoice::AllDirections) {
                  success1 = applyAllDirectionsInterconnectToCopy(clonedCopyOp1, 
                                                                  choice1.horizontal, 
                                                                  choice1.vertical, 
                                                                  module, builder);
                } else if (choice1.type == InterconnectChoice::Single) {
                  loom::df::InterconnectsOp op1 = choice1.horizontal ? choice1.horizontal : choice1.vertical;
                  success1 = applyInterconnectAndBroadcastToCopy(clonedCopyOp1, op1, module, builder);
                } else { // DRAM
                  success1 = applyInterconnectAndBroadcastToCopy(clonedCopyOp1, nullptr, module, builder);
                }
                
                if (!success1) {
                  return failure();
                }

                loom::CopyOp clonedCopyOp2 = clonedCopyOps[1];
                builder.setInsertionPoint(clonedCopyOp2);
                
                // Apply second interconnect choice
                bool success2 = false;
                if (choice2.type == InterconnectChoice::AllDirections) {
                  success2 = applyAllDirectionsInterconnectToCopy(clonedCopyOp2, 
                                                                  choice2.horizontal, 
                                                                  choice2.vertical, 
                                                                  module, builder);
                } else if (choice2.type == InterconnectChoice::Single) {
                  loom::df::InterconnectsOp op2 = choice2.horizontal ? choice2.horizontal : choice2.vertical;
                  success2 = applyInterconnectAndBroadcastToCopy(clonedCopyOp2, op2, module, builder);
                } else { // DRAM
                  success2 = applyInterconnectAndBroadcastToCopy(clonedCopyOp2, nullptr, module, builder);
                }
                
                if (!success2) {
                  return failure();
                }
                
                return success();
              },
              insertAfter);
          
          if (clonedFunc) {
            // Update insertion point to the wrapper module
            ModuleOp clonedParentModule = loom::utils::getParentModule(clonedFunc);
            if (clonedParentModule) {
              insertAfter = clonedParentModule;
            }
          }
        }
      }
      
      // Erase the original function's parent module (which contains the original function)
      if (parentModule) {
        parentModule.erase();
      } else {
        originalFunc.erase();
      }
    }
  }

  /// Flag indicating whether this pass should only perform analysis (currently unused).
  bool analysisOnly = false;
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createEnumerateCopyBroadcastPass(bool analysisOnly) {
  return std::make_unique<EnumerateCopyBroadcastPass>(analysisOnly);
}

void loom::passes::registerEnumerateCopyBroadcastPass() {
  PassRegistration<EnumerateCopyBroadcastPass>();
}
