/**
 * @file constraint_factorize.cpp
 * @brief Implementation of polynomial constraint factorization pass.
 */

#include "constraint_factorize.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#include <algorithm>

#define DEBUG_TYPE "loom-constraint-factorize"

namespace loom {
namespace constraint_opt {

using namespace mlir;

namespace {

/// Parsed monomial for factorization
struct ParsedMonomial {
  int64_t coeff;
  llvm::SmallVector<int64_t, 4> vars;

  size_t degree() const { return vars.size(); }

  bool hasVar(int64_t v) const {
    return std::find(vars.begin(), vars.end(), v) != vars.end();
  }

  void removeVar(int64_t v) {
    vars.erase(std::remove(vars.begin(), vars.end(), v), vars.end());
  }
};

/// Parse monomials from attribute
llvm::SmallVector<ParsedMonomial> parseMonomials(ArrayAttr attr) {
  llvm::SmallVector<ParsedMonomial> result;
  for (auto mAttr : attr) {
    auto dict = cast<DictionaryAttr>(mAttr);
    ParsedMonomial m;
    m.coeff = cast<IntegerAttr>(dict.get("coeff")).getInt();
    for (auto v : cast<ArrayAttr>(dict.get("vars"))) {
      m.vars.push_back(cast<IntegerAttr>(v).getInt());
    }
    result.push_back(std::move(m));
  }
  return result;
}

/// Build monomials attribute
ArrayAttr buildMonomialsAttr(MLIRContext *ctx,
                             llvm::ArrayRef<ParsedMonomial> monomials) {
  OpBuilder builder(ctx);
  llvm::SmallVector<Attribute, 8> attrs;
  for (const auto &m : monomials) {
    llvm::SmallVector<NamedAttribute, 2> namedAttrs;
    namedAttrs.push_back(
        builder.getNamedAttr("coeff", builder.getI64IntegerAttr(m.coeff)));
    namedAttrs.push_back(
        builder.getNamedAttr("vars", builder.getI64ArrayAttr(m.vars)));
    attrs.push_back(DictionaryAttr::get(ctx, namedAttrs));
  }
  return builder.getArrayAttr(attrs);
}

/// Find the best common factor using greedy frequency counting
/// Returns -1 if no common factor found
int64_t findBestFactor(llvm::ArrayRef<ParsedMonomial> monomials) {
  // Only consider monomials with degree >= 2
  llvm::SmallVector<const ParsedMonomial *> nonLinear;
  for (const auto &m : monomials) {
    if (m.degree() >= 2) {
      nonLinear.push_back(&m);
    }
  }

  if (nonLinear.size() < 2)
    return -1; // Need at least 2 non-linear terms to factor

  // Count frequency of each variable in non-linear terms
  llvm::DenseMap<int64_t, int64_t> freq;
  for (const auto *m : nonLinear) {
    for (int64_t v : m->vars) {
      freq[v]++;
    }
  }

  // Find variable with highest score = freq * (number of terms it appears in)
  // We want a variable that appears in all or most non-linear terms
  int64_t bestVar = -1;
  int64_t bestScore = 0;

  for (const auto &[var, count] : freq) {
    // Factor must appear in at least 2 terms
    if (count >= 2) {
      // Score is just frequency for now (could weight by degree)
      int64_t score = count;
      if (score > bestScore) {
        bestScore = score;
        bestVar = var;
      }
    }
  }

  return bestVar;
}

/// Factorize a single polynomial constraint
/// Returns true if factorization was applied
bool factorizePolynomialConstraint(PolynomialConstraintOp pcOp,
                                   ConstraintSpaceOp csOp) {
  MLIRContext *ctx = pcOp.getContext();
  auto monomials = parseMonomials(pcOp.getMonomials());

  int64_t factorVar = findBestFactor(monomials);
  if (factorVar < 0)
    return false;

  LLVM_DEBUG(llvm::dbgs() << "Factoring out variable index " << factorVar
                          << "\n");

  // Separate monomials into:
  // - factorable: contain the factor variable
  // - remaining: do not contain the factor variable
  llvm::SmallVector<ParsedMonomial> factorable, remaining;
  for (auto &m : monomials) {
    if (m.hasVar(factorVar)) {
      factorable.push_back(m);
    } else {
      remaining.push_back(m);
    }
  }

  if (factorable.size() < 2)
    return false;

  // Check if all factorable terms reduce to linear after factoring
  bool allReduceToLinear = true;
  for (auto &m : factorable) {
    if (m.degree() > 2) { // Will become degree > 1 after factoring
      allReduceToLinear = false;
      break;
    }
  }

  if (!allReduceToLinear) {
    // For now, only handle cases where factoring produces linear quotients
    // More complex cases would need recursive handling
    return false;
  }

  // Create the expression: sum of quotient terms
  // E.g., for MK + NK, factor K gives expression M + N
  OpBuilder builder(csOp.getBodyBlock(), csOp.getBodyBlock()->end());
  builder.setInsertionPoint(pcOp);

  // Collect the quotient terms (what's left after removing the factor)
  // Each quotient should be a single variable
  llvm::SmallVector<Value> exprOperands;
  llvm::SmallVector<int64_t> exprCoeffs;

  auto pcOperands = pcOp.getOperands();

  for (auto &m : factorable) {
    m.removeVar(factorVar);
    if (m.vars.size() == 1) {
      // Single variable quotient
      exprOperands.push_back(pcOperands[m.vars[0]]);
      exprCoeffs.push_back(m.coeff);
    } else if (m.vars.empty()) {
      // Constant quotient (just coefficient) - factor from linear term
      // This means original was c*K, quotient is c
      // We'll handle this by creating a constant-weighted term
      // For now, skip this case
      return false;
    } else {
      // Multi-variable quotient - too complex for now
      return false;
    }
  }

  if (exprOperands.empty())
    return false;

  // Create loom.expression op for the quotient sum
  auto exprOp = builder.create<ExpressionOp>(
      pcOp.getLoc(), builder.getIndexType(), exprOperands,
      builder.getI64ArrayAttr(exprCoeffs));

  // Build new monomials: factor_var * expression, plus remaining terms
  llvm::SmallVector<ParsedMonomial> newMonomials = remaining;

  // The factored term becomes: coeff=1, vars=[factor_var, expr_idx]
  // We need to add the expression result as a new operand
  int64_t exprIdx = pcOp.getOperands().size(); // Index of the new operand

  ParsedMonomial factoredTerm;
  factoredTerm.coeff = 1;
  factoredTerm.vars = {factorVar, exprIdx};
  newMonomials.push_back(factoredTerm);

  // Create new operands list including the expression result
  llvm::SmallVector<Value> newOperands(pcOp.getOperands().begin(),
                                       pcOp.getOperands().end());
  newOperands.push_back(exprOp.getResult());

  // Replace the polynomial constraint with updated version
  auto newPcOp = builder.create<PolynomialConstraintOp>(
      pcOp.getLoc(), newOperands, buildMonomialsAttr(ctx, newMonomials),
      pcOp.getUpperBoundAttr());

  pcOp.erase();

  LLVM_DEBUG(llvm::dbgs() << "Created factored constraint with expression\n");

  return true;
}

} // namespace

LogicalResult runConstraintFactorize(ModuleOp module) {
  bool changed = true;
  int iterations = 0;
  const int maxIterations = 10; // Prevent infinite loops

  while (changed && iterations < maxIterations) {
    changed = false;
    iterations++;

    module.walk([&](ConstraintSpaceOp csOp) {
      // Collect ops to process (avoid modifying while iterating)
      llvm::SmallVector<PolynomialConstraintOp> pcOps;
      for (auto &op : *csOp.getBodyBlock()) {
        if (auto pcOp = dyn_cast<PolynomialConstraintOp>(&op)) {
          pcOps.push_back(pcOp);
        }
      }

      for (auto pcOp : pcOps) {
        if (factorizePolynomialConstraint(pcOp, csOp)) {
          changed = true;
        }
      }
    });
  }

  LLVM_DEBUG(llvm::dbgs() << "Factorization completed in " << iterations
                          << " iterations\n");

  return success();
}

} // namespace constraint_opt
} // namespace loom
