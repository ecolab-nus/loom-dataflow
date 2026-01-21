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

#include <algorithm>

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
                          << "' with "
                          << clonedSpace.getBodyBlock()->getOperations().size()
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

  LLVM_DEBUG(llvm::dbgs() << "Added linear constraint with " << varNames.size()
                          << " variables\n");

  return lcOp;
}

llvm::SmallVector<Monomial> parseMonomials(ArrayAttr monomialsAttr) {
  llvm::SmallVector<Monomial> result;
  if (!monomialsAttr)
    return result;

  for (auto mAttr : monomialsAttr) {
    auto dict = dyn_cast<DictionaryAttr>(mAttr);
    if (!dict)
      continue;

    Monomial m;
    auto coeffAttr = dict.getAs<IntegerAttr>("coeff");
    auto varsAttr = dict.getAs<ArrayAttr>("vars");

    if (coeffAttr && varsAttr) {
      m.coeff = coeffAttr.getInt();
      for (auto v : varsAttr) {
        if (auto vInt = dyn_cast<IntegerAttr>(v))
          m.varIndices.push_back(vInt.getInt());
      }
      // Sort variables within monomial to maintain canonical form
      std::sort(m.varIndices.begin(), m.varIndices.end());
      result.push_back(std::move(m));
    }
  }
  return result;
}

ArrayAttr buildMonomialsAttr(MLIRContext *ctx,
                             llvm::ArrayRef<Monomial> monomials) {
  OpBuilder builder(ctx);
  llvm::SmallVector<Attribute, 8> monomialAttrs;
  for (const auto &m : monomials) {
    llvm::SmallVector<NamedAttribute, 2> attrs;
    attrs.push_back(
        builder.getNamedAttr("coeff", builder.getI64IntegerAttr(m.coeff)));
    attrs.push_back(
        builder.getNamedAttr("vars", builder.getI64ArrayAttr(m.varIndices)));
    monomialAttrs.push_back(DictionaryAttr::get(ctx, attrs));
  }
  return builder.getArrayAttr(monomialAttrs);
}

int64_t gcd(int64_t a, int64_t b) {
  a = std::abs(a);
  b = std::abs(b);
  while (b != 0) {
    int64_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

int64_t gcdVector(llvm::ArrayRef<int64_t> values) {
  if (values.empty())
    return 1;
  int64_t result = values[0];
  for (size_t i = 1; i < values.size(); ++i) {
    result = gcd(result, values[i]);
    if (result == 1)
      return 1;
  }
  return result;
}

PolynomialConstraintOp addPolynomialConstraint(
    ConstraintSpaceOp csOp, llvm::ArrayRef<llvm::StringRef> varNames,
    llvm::ArrayRef<Monomial> monomials, int64_t upperBound) {
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
  MLIRContext *ctx = csOp.getContext();

  // Create the polynomial constraint op
  auto pcOp = builder.create<PolynomialConstraintOp>(
      csOp.getLoc(), operands, buildMonomialsAttr(ctx, monomials),
      builder.getI64IntegerAttr(upperBound));

  LLVM_DEBUG(llvm::dbgs() << "Added polynomial constraint with "
                          << monomials.size() << " monomials\n");

  return pcOp;
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

  LLVM_DEBUG(llvm::dbgs() << "Added range constraint for '" << varName << "': ["
                          << lowerBound << ", " << upperBound << "]\n");

  return rangeOp;
}

void updateRangeLowerBounds(ConstraintSpaceOp csOp, int64_t newLowerBound) {
  for (Operation &op : csOp.getBodyBlock()->getOperations()) {
    if (auto rangeOp = dyn_cast<RangeOp>(&op)) {
      if (static_cast<int64_t>(rangeOp.getLowerBound()) < newLowerBound) {
        rangeOp.setLowerBound(newLowerBound);
      }
    }
  }
}

void addAlignConstraintsForAllVars(ConstraintSpaceOp csOp, int64_t alignment) {
  OpBuilder builder(csOp.getBodyBlock(), csOp.getBodyBlock()->end());
  for (Operation &op : csOp.getBodyBlock()->getOperations()) {
    if (auto symVar = dyn_cast<SymbolicVarOp>(&op)) {
      builder.create<AlignOp>(csOp.getLoc(), symVar.getResult(), alignment);
    }
  }
}

PolynomialConstraintOp
addPipelineParallelismConstraint(ConstraintSpaceOp csOp,
                                 llvm::ArrayRef<llvm::StringRef> varNames,
                                 int64_t minMatUnits, int64_t alignment) {
  llvm::SmallVector<Monomial> monomials;
  Monomial m;
  m.coeff = -1; // -M*N*K
  for (unsigned i = 0; i < varNames.size(); ++i) {
    m.varIndices.push_back(i);
  }
  monomials.push_back(m);

  // minMatUnits * alignment^3
  int64_t lowerBound = minMatUnits * alignment * alignment * alignment;
  // -M*N*K <= -lowerBound
  return addPolynomialConstraint(csOp, varNames, monomials, -lowerBound);
}

bool isFeasible(ConstraintSpaceOp csOp) {
  // Build the constraint set from IR
  ConstraintSet cs = AnalysisEngine::buildConstraintSet(csOp);

  // Check if empty (infeasible)
  bool feasible = !cs.isEmpty();

  LLVM_DEBUG({
    llvm::dbgs() << "Feasibility check for constraint space '"
                 << csOp.getSymName()
                 << "': " << (feasible ? "FEASIBLE" : "INFEASIBLE") << "\n";
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
