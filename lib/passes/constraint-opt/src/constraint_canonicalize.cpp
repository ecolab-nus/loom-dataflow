#include "Passes.h"
#include "constraint_space_utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#include <algorithm>

#define DEBUG_TYPE "loom-constraint-canonicalize"

namespace loom {
namespace constraint_opt {

#define GEN_PASS_DEF_LOOMCONSTRAINTCANONICALIZE
#include "Passes.h.inc"

using namespace mlir;
using namespace loom::lcs;

namespace {

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

  llvm::SmallVector<Monomial> combined;
  for (const auto &m : monomials) {
    if (!combined.empty() && combined.back().sameVars(m)) {
      combined.back().coeff += m.coeff;
    } else {
      combined.push_back(m);
    }
  }

  // Remove zero-coefficient terms
  combined.erase(std::remove_if(combined.begin(), combined.end(),
                                [](const Monomial &m) { return m.coeff == 0; }),
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

struct LoomConstraintCanonicalize
    : public impl::LoomConstraintCanonicalizeBase<LoomConstraintCanonicalize> {
  using LoomConstraintCanonicalizeBase::LoomConstraintCanonicalizeBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](ConstraintSpaceOp csOp) {
      for (auto &op : *csOp.getBodyBlock()) {
        if (auto pcOp = dyn_cast<PolynomialConstraintOp>(&op)) {
          if (failed(canonicalizePolynomialConstraint(pcOp))) {
            signalPassFailure();
          }
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLoomConstraintCanonicalizePass() {
  return std::make_unique<LoomConstraintCanonicalize>();
}

} // namespace constraint_opt
} // namespace loom
