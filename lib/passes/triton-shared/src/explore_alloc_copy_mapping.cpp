/**
 * @file explore_alloc_copy_mapping.cpp
 * @brief Implementation for alloc/copy mapping exploration.
 * @details
 * This pass analyzes loom.copy operations and checks their source operations
 * for spatial reuse information from loom.reinterpret_cast operations.
 */

#include "explore_alloc_copy_mapping.h"
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

using namespace mlir;

namespace {

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
 * @brief Find all candidate df.interconnects operations for a loom.copy operation.
 * @details This function checks if the copy operation's source has spatial reuse,
 * and if so, finds all df.interconnects operations that match the spatial dimensions
 * that the reinterpret_cast depends on.
 * @param copyOp The loom.copy operation to analyze.
 * @param module The module containing the df.interconnects operations.
 * @return A vector of candidate df.interconnects operations (empty if no matches).
 */
static SmallVector<Operation *> findCandidateInterconnects(Operation *copyOp,
                                                            ModuleOp module) {
  // Add nullptr to list to represent DRAM option (no interconnect)
  SmallVector<Operation *> candidates;
  candidates.push_back(nullptr); // DRAM option

  // Verify this is a loom.copy operation
  if (copyOp->getName().getStringRef() != "loom.copy")
    return candidates;

  // Get the source operand (first operand)
  if (copyOp->getNumOperands() < 1)
    return candidates;
  
  Value src = copyOp->getOperand(0);
  
  // Check if the source is defined by a loom.reinterpret_cast
  Operation *srcDef = src.getDefiningOp();
  if (!srcDef)
    return candidates;
  
  if (srcDef->getName().getStringRef() != "loom.reinterpret_cast")
    return candidates;
  
  // Get the spatial_reuse attribute
  auto spatialReuseAttr = srcDef->getAttrOfType<BoolAttr>("spatial_reuse");
  if (!spatialReuseAttr || !spatialReuseAttr.getValue())
    return candidates;
  
  // Get offsets from reinterpret_cast (similar to reinterpret_cast_reuse.cpp)
  auto segmentSizes = srcDef->getAttrOfType<DenseI32ArrayAttr>("operand_segment_sizes");
  if (!segmentSizes || segmentSizes.size() < 4)
    return candidates;
  
  unsigned sourceSize = segmentSizes[0];
  unsigned offsetsSize = segmentSizes[1];
  unsigned offsetsStart = sourceSize;
  
  // Collect dynamic offsets
  SmallVector<Value, 4> dynamicOffsets;
  for (unsigned i = 0; i < offsetsSize; ++i) {
    Value offsetVal = srcDef->getOperand(offsetsStart + i);
    dynamicOffsets.push_back(offsetVal);
  }

  // Collect enclosing affine.parallel loops
  SmallVector<affine::AffineParallelOp, 4> parallelLoops;
  for (Operation *parent = copyOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto par = dyn_cast<affine::AffineParallelOp>(parent))
      parallelLoops.push_back(par);
  }

  // For each parallel loop, check if offsets depend on its IVs
  for (auto par : parallelLoops) {
    // Get loom.mapped_to attribute
    auto mappedToAttr = par->getAttrOfType<SymbolRefAttr>("loom.mapped_to");
    if (!mappedToAttr)
      continue;
    
    // Check if any dynamic offset depends on any IV of this parallel loop
    bool hasDependency = false;
    for (Value iv : par.getIVs()) {
      for (Value offset : dynamicOffsets) {
        if (dependsOn(offset, iv)) {
          hasDependency = true;
          break;
        }
      }
      if (hasDependency)
        break;
    }
    
    if (!hasDependency)
      continue;

    // Find df.interconnects operations in module with matching spatial_dims
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "df.interconnects")
        return;
      
      // Get spatial_dims attribute (ArrayAttr of SymbolRefAttr)
      auto spatialDimsAttr = op->getAttrOfType<ArrayAttr>("spatial_dims");
      if (!spatialDimsAttr)
        return;
      
      // Check if spatial_dims contains the mapped_to symbol
      for (Attribute attr : spatialDimsAttr) {
        auto symbolRef = llvm::dyn_cast<SymbolRefAttr>(attr);
        if (symbolRef && symbolRef == mappedToAttr) {
          candidates.push_back(op);
          return; // Found match, no need to check further
        }
      }
    });
  }

  return candidates;
}

/**
 * @brief Apply an interconnect to a loom.copy operation.
 * @details Adds the interconnect operation's result to the copy operation's
 * interconnect operands. If interconnectOp is nullptr, leaves interconnect empty
 * (indicating DRAM access).
 * 
 * Note: There's a type mismatch between df.interconnects (returns InterconnectHandleType)
 * and loom.copy's interconnect operands (expects Variadic<Index>). We handle this by
 * attempting to cast the interconnect result to Index type, or using it directly if
 * the type system allows.
 * 
 * @param copyOp The loom.copy operation to modify.
 * @param interconnectOp The interconnect operation to apply (can be nullptr for DRAM).
 * @param builder OpBuilder for creating new operations.
 * @return true if successfully applied, false otherwise.
 */
static bool applyInterconnectToCopy(Operation *copyOp, Operation *interconnectOp,
                                    OpBuilder &builder) {
  // Precondition: copyOp is loom.copy with format:
  // loom.copy %src, %dst, interconnect : [], broadcast : [1, 1]
  // interconnect is now an attribute (ArrayAttr of SymbolRefAttr), not an operand
  // Since CopyOp no longer uses AttrSizedOperandSegments, operands are just: src, dst, optional provenance

  // Extract current operands: src and dst are always the first two operands
  Value src = copyOp->getOperand(0);
  Value dst = copyOp->getOperand(1);
  
  // Check if provenance exists (it would be the third operand if present)
  Value provenance = (copyOp->getNumOperands() > 2) ? copyOp->getOperand(2) : Value();
  
  // Get broadcast attribute
  auto broadcastAttr = copyOp->getAttrOfType<DenseI64ArrayAttr>("broadcast");

  // Build new operands: src, dst, optional provenance (no interconnect operands)
  SmallVector<Value, 3> newOperands;
  newOperands.push_back(src);
  newOperands.push_back(dst);
  if (provenance)
    newOperands.push_back(provenance);

  // Rebuild the operation
  builder.setInsertionPoint(copyOp);
  OperationState state(copyOp->getLoc(), "loom.copy");
  state.addOperands(newOperands);
  if (broadcastAttr)
    state.addAttribute("broadcast", broadcastAttr);

  // Add interconnect attribute if provided
  if (interconnectOp) {
    // Get the symbol name from the interconnect operation
    auto symNameAttr = interconnectOp->getAttrOfType<StringAttr>("sym_name");
    if (symNameAttr) {
      // Create a SymbolRefAttr pointing to the interconnect symbol
      auto symbolRef = SymbolRefAttr::get(builder.getContext(), symNameAttr.getValue());
      // Create ArrayAttr with the symbol reference
      auto interconnectAttr = ArrayAttr::get(builder.getContext(), {symbolRef});
      state.addAttribute("interconnect", interconnectAttr);
    } else {
      // Cannot get symbol name, skip interconnect
      return false;
    }
  }
  // If interconnectOp is nullptr (DRAM), we don't add the interconnect attribute

  // Copy other attributes (excluding broadcast and interconnect, which we handle separately)
  for (auto &attr : copyOp->getAttrs()) {
    if (attr.getName() != "broadcast" && attr.getName() != "interconnect") {
      state.addAttribute(attr.getName(), attr.getValue());
    }
  }

  builder.create(state);
  // Note: loom.copy has no results, so we don't need replaceAllUsesWith
  copyOp->erase();
  
  return true;
}

/**
 * @brief Find all loom.copy operations in a function.
 * @param func The function to search.
 * @return A vector of loom.copy operations.
 */
static SmallVector<Operation *> findCopyOpsInFunc(func::FuncOp func) {
  SmallVector<Operation *> copyOps;
  func.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "loom.copy")
      copyOps.push_back(op);
  });
  return copyOps;
}

/**
 * @brief Generate a function name based on interconnect choices.
 * @param baseName The base function name.
 * @param interconnect1 The first interconnect operation (can be nullptr for DRAM).
 * @param interconnect2 The second interconnect operation (can be nullptr for DRAM).
 * @return The new function name.
 */
static std::string generateFunctionName(StringRef baseName,
                                        Operation *interconnect1,
                                        Operation *interconnect2) {
  std::string newName = baseName.str();
  
  // Get interconnect labels if available
  auto getInterconnectLabel = [](Operation *op) -> std::string {
    if (!op)
      return "dram";
    auto labelAttr = op->getAttrOfType<StringAttr>("label");
    if (labelAttr)
      return labelAttr.getValue().str();
    return "interconnect";
  };
  
  std::string label1 = getInterconnectLabel(interconnect1);
  std::string label2 = getInterconnectLabel(interconnect2);
  
  newName += "__" + label1 + "_" + label2;
  
  return newName;
}

struct ExploreAllocCopyMappingPass
    : public PassWrapper<ExploreAllocCopyMappingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExploreAllocCopyMappingPass)

  ExploreAllocCopyMappingPass() = default;
  ExploreAllocCopyMappingPass(bool analysisOnly) : analysisOnly(analysisOnly) {}

  StringRef getArgument() const override {
    return "loom-explore-alloc-copy-mapping";
  }
  StringRef getDescription() const override {
    return "Check spatial reuse for loom.copy operations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder moduleBuilder(module.getBodyRegion());

    // Collect all functions first to avoid iterator invalidation
    SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>().begin(),
                                     module.getOps<func::FuncOp>().end());

    for (func::FuncOp originalFunc : funcs) {
      // Find the two loom.copy operations in this function
      SmallVector<Operation *> copyOps = findCopyOpsInFunc(originalFunc);
      if (copyOps.size() != 2) {
        // Skip if not exactly 2 copy operations
        continue;
      }

      Operation *copyOp1 = copyOps[0];
      Operation *copyOp2 = copyOps[1];

      // Find candidates for each copy (empty list means DRAM option)
      SmallVector<Operation *> candidates1 = findCandidateInterconnects(copyOp1, module);
      SmallVector<Operation *> candidates2 = findCandidateInterconnects(copyOp2, module);

      // Generate cross product of all combinations
      Operation *insertAfter = originalFunc;
      for (Operation *interconnect1 : candidates1) {
        for (Operation *interconnect2 : candidates2) {
          // Clone the function
          moduleBuilder.setInsertionPointAfter(insertAfter);
          IRMapping map;
          func::FuncOp clonedFunc = cast<func::FuncOp>(
              moduleBuilder.clone(*originalFunc, map));

          // Find the corresponding copy operations in the cloned function
          // Since clone preserves structure, walk order should match
          SmallVector<Operation *> clonedCopyOps = findCopyOpsInFunc(clonedFunc);
          if (clonedCopyOps.size() != 2) {
            clonedFunc.erase();
            continue;
          }
          
          Operation *clonedCopyOp1 = clonedCopyOps[0];

          // Apply interconnects to the cloned copies
          // Create builder with proper insertion point in the cloned function
          OpBuilder builder(clonedCopyOp1);
          
          // Apply to first copy
          if (!applyInterconnectToCopy(clonedCopyOp1, interconnect1, builder)) {
            clonedFunc.erase();
            continue;
          }

          Operation *clonedCopyOp2After = clonedCopyOps[1];
          builder.setInsertionPoint(clonedCopyOp2After);
          if (!applyInterconnectToCopy(clonedCopyOp2After, interconnect2, builder)) {
            clonedFunc.erase();
            continue;
          }

          // Generate and set new function name
          std::string newName = generateFunctionName(originalFunc.getSymName(),
                                                      interconnect1, interconnect2);
          clonedFunc.setName(newName);

          // Update insertion point for next clone
          insertAfter = clonedFunc;
        }
      }
    }
  }

  bool analysisOnly = false;
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createExploreAllocCopyMappingPass(bool analysisOnly) {
  return std::make_unique<ExploreAllocCopyMappingPass>(analysisOnly);
}

void loom::passes::registerExploreAllocCopyMappingPass() {
  PassRegistration<ExploreAllocCopyMappingPass>();
}
