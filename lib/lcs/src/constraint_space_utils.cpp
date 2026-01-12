//===- constraint_space_utils.cpp - Constraint Space Utilities -----------===//
//
// Implementation of utilities for manipulating Loom ConstraintSpaceOp.
//
//===----------------------------------------------------------------------===//

#include "constraint_space_utils.h"
#include "analysis_engine.h"
#include "constraint_set.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

// Include the generated Loom dialect headers
#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#define DEBUG_TYPE "loom-constraint-space-utils"

namespace loom {
namespace lcs {

using namespace mlir;

ConstraintSpaceOp cloneConstraintSpace(OpBuilder &builder,
                                       ConstraintSpaceOp sourceSpace,
                                       llvm::StringRef newName) {
  // Determine the name for the cloned constraint space
  llvm::StringRef spaceName =
      newName.empty() ? sourceSpace.getSymName() : newName;

  // Clone the entire constraint space operation
  IRMapping mapping;
  auto clonedSpace =
      cast<ConstraintSpaceOp>(builder.clone(*sourceSpace, mapping));

  // Update the name if a new one was provided
  if (!newName.empty()) {
    clonedSpace.setSymName(newName);
  }

  LLVM_DEBUG(llvm::dbgs() << "Cloned constraint space '" << spaceName
                          << "' with " << clonedSpace.getBodyBlock()->getOperations().size()
                          << " operations\n");

  return clonedSpace;
}

ConstraintSpaceOp findConstraintSpace(ModuleOp module) {
  ConstraintSpaceOp result = nullptr;

  module.walk([&](ConstraintSpaceOp csOp) {
    if (!result) {
      result = csOp;
    }
  });

  return result;
}

SymbolicVarOp findSymbolicVar(ConstraintSpaceOp csOp, llvm::StringRef varName) {
  for (Operation &op : csOp.getBodyBlock()->getOperations()) {
    if (auto symVar = dyn_cast<SymbolicVarOp>(&op)) {
      if (symVar.getName() == varName) {
        return symVar;
      }
    }
  }
  return nullptr;
}

LinearConstraintOp addLinearConstraint(ConstraintSpaceOp csOp,
                                       llvm::ArrayRef<llvm::StringRef> varNames,
                                       AffineMap constraintMap) {
  // Find all the symbolic variables
  llvm::SmallVector<Value, 4> operands;
  for (llvm::StringRef name : varNames) {
    SymbolicVarOp symVar = findSymbolicVar(csOp, name);
    if (!symVar) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Could not find symbolic variable '" << name << "'\n");
      return nullptr;
    }
    operands.push_back(symVar.getResult());
  }

  // Insert at the end of the constraint space body
  OpBuilder builder(csOp.getBodyBlock(), csOp.getBodyBlock()->end());

  // Create the linear constraint op
  auto lcOp = builder.create<LinearConstraintOp>(
      csOp.getLoc(), operands, AffineMapAttr::get(constraintMap));

  LLVM_DEBUG(llvm::dbgs() << "Added linear constraint with "
                          << varNames.size() << " variables\n");

  return lcOp;
}

RangeOp addRangeConstraint(ConstraintSpaceOp csOp, llvm::StringRef varName,
                           int64_t lowerBound, int64_t upperBound) {
  // Find the symbolic variable
  SymbolicVarOp symVar = findSymbolicVar(csOp, varName);
  if (!symVar) {
    LLVM_DEBUG(llvm::dbgs()
               << "Could not find symbolic variable '" << varName << "'\n");
    return nullptr;
  }

  // Insert at the end of the constraint space body
  OpBuilder builder(csOp.getBodyBlock(), csOp.getBodyBlock()->end());

  // Create the range op
  auto rangeOp = builder.create<RangeOp>(csOp.getLoc(), symVar.getResult(),
                                          lowerBound, upperBound);

  LLVM_DEBUG(llvm::dbgs() << "Added range constraint for '" << varName
                          << "': [" << lowerBound << ", " << upperBound << "]\n");

  return rangeOp;
}

bool isFeasible(ConstraintSpaceOp csOp) {
  // Build the constraint set from IR
  ConstraintSet cs = AnalysisEngine::buildConstraintSet(csOp);

  // Check if empty (infeasible)
  bool feasible = !cs.isEmpty();

  LLVM_DEBUG({
    llvm::dbgs() << "Feasibility check for constraint space '"
                 << csOp.getSymName() << "': "
                 << (feasible ? "FEASIBLE" : "INFEASIBLE") << "\n";
    if (!feasible) {
      cs.dump();
    }
  });

  return feasible;
}

void setPassNameAttr(ModuleOp module, llvm::StringRef passName) {
  module->setAttr("loom.pass_name",
                  StringAttr::get(module.getContext(), passName));
}

llvm::StringRef getPassNameAttr(ModuleOp module) {
  if (auto attr = module->getAttrOfType<StringAttr>("loom.pass_name")) {
    return attr.getValue();
  }
  return "";
}

} // namespace lcs
} // namespace loom
