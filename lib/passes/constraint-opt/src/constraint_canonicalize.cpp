/**
 * @file constraint_canonicalize.cpp
 * @brief Implementation of polynomial constraint canonicalization pass.
 */

#include "constraint_canonicalize.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#include <algorithm>
#include <numeric>

#define DEBUG_TYPE "loom-constraint-canonicalize"

namespace loom {
namespace constraint_opt {

using namespace mlir;

namespace {

/// Compute GCD of two integers
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

/// Compute GCD of a vector of integers
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

/// Parsed monomial structure for manipulation
struct ParsedMonomial {
  int64_t coeff;
  llvm::SmallVector<int64_t, 4> vars; // Sorted variable indices

  bool operator<(const ParsedMonomial &other) const {
    if (vars.size() != other.vars.size())
      return vars.size() < other.vars.size();
    return vars < other.vars;
  }

  bool sameVars(const ParsedMonomial &other) const {
    return vars == other.vars;
  }
};

/// Parse monomials from attribute
llvm::SmallVector<ParsedMonomial> parseMonomials(ArrayAttr monomialsAttr) {
  llvm::SmallVector<ParsedMonomial> result;
  for (auto mAttr : monomialsAttr) {
    auto dict = cast<DictionaryAttr>(mAttr);
    ParsedMonomial m;
    m.coeff = cast<IntegerAttr>(dict.get("coeff")).getInt();
    auto varsArr = cast<ArrayAttr>(dict.get("vars"));
    for (auto v : varsArr) {
      m.vars.push_back(cast<IntegerAttr>(v).getInt());
    }
    // Sort vars within monomial
    std::sort(m.vars.begin(), m.vars.end());
    result.push_back(std::move(m));
  }
  return result;
}

/// Build monomials attribute from parsed structure
ArrayAttr buildMonomialsAttr(MLIRContext *ctx,
                             llvm::ArrayRef<ParsedMonomial> monomials) {
  OpBuilder builder(ctx);
  llvm::SmallVector<Attribute, 8> monomialAttrs;
  for (const auto &m : monomials) {
    llvm::SmallVector<NamedAttribute, 2> attrs;
    attrs.push_back(
        builder.getNamedAttr("coeff", builder.getI64IntegerAttr(m.coeff)));
    attrs.push_back(
        builder.getNamedAttr("vars", builder.getI64ArrayAttr(m.vars)));
    monomialAttrs.push_back(DictionaryAttr::get(ctx, attrs));
  }
  return builder.getArrayAttr(monomialAttrs);
}

/// Canonicalize a single polynomial constraint
LogicalResult canonicalizePolynomialConstraint(PolynomialConstraintOp pcOp) {
  MLIRContext *ctx = pcOp.getContext();

  // 1. Parse monomials
  auto monomials = parseMonomials(pcOp.getMonomials());
  if (monomials.empty())
    return success();

  int64_t upperBound = pcOp.getUpperBound();

  // 2. Sort variables within each monomial (already done in parseMonomials)

  // 3. Sort monomials and combine like terms
  std::sort(monomials.begin(), monomials.end());

  llvm::SmallVector<ParsedMonomial> combined;
  for (const auto &m : monomials) {
    if (!combined.empty() && combined.back().sameVars(m)) {
      combined.back().coeff += m.coeff;
    } else {
      combined.push_back(m);
    }
  }

  // Remove zero-coefficient terms
  combined.erase(
      std::remove_if(combined.begin(), combined.end(),
                     [](const ParsedMonomial &m) { return m.coeff == 0; }),
      combined.end());

  if (combined.empty()) {
    // Constraint is trivially satisfied (0 <= upperBound)
    // Keep as-is for now, could optimize differently
    return success();
  }

  // 4. GCD reduction
  llvm::SmallVector<int64_t> allValues;
  for (const auto &m : combined) {
    allValues.push_back(m.coeff);
  }
  allValues.push_back(upperBound);

  int64_t g = gcdVector(allValues);
  if (g > 1) {
    for (auto &m : combined) {
      m.coeff /= g;
    }
    upperBound /= g;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Canonicalized constraint: GCD=" << g << ", "
                 << combined.size() << " monomials\n";
  });

  // 5. Update the operation
  pcOp.setMonomialsAttr(buildMonomialsAttr(ctx, combined));
  pcOp.setUpperBoundAttr(
      IntegerAttr::get(IntegerType::get(ctx, 64), upperBound));

  return success();
}

} // namespace

LogicalResult runConstraintCanonicalize(ModuleOp module) {
  LogicalResult result = success();

  module.walk([&](ConstraintSpaceOp csOp) {
    for (auto &op : *csOp.getBodyBlock()) {
      if (auto pcOp = dyn_cast<PolynomialConstraintOp>(&op)) {
        if (failed(canonicalizePolynomialConstraint(pcOp))) {
          result = failure();
        }
      }
    }
  });

  return result;
}

} // namespace constraint_opt
} // namespace loom
