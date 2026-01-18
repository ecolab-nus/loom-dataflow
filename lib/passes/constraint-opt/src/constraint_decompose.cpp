#include "Passes.h"
#include "constraint_space_utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#define DEBUG_TYPE "loom-constraint-decompose"

namespace loom {
namespace constraint_opt {

#define GEN_PASS_DEF_LOOMCONSTRAINTDECOMPOSE
#include "Passes.h.inc"

using namespace mlir;
using namespace loom::lcs;

namespace {

/// Helper to get or create a multiplication expression
Value getOrMulExpression(
    OpBuilder &builder, Location loc, Value lhs, Value rhs,
    llvm::DenseMap<std::pair<Value, Value>, Value> &mulCache) {
  // Sort operands to handle commutativity if they are standard values,
  // but SSA values don't have a stable order easily. We use pointer comparison.
  std::pair<Value, Value> key = {lhs, rhs};
  if (lhs.getAsOpaquePointer() > rhs.getAsOpaquePointer()) {
    key = {rhs, lhs};
  }

  if (mulCache.count(key)) {
    return mulCache[key];
  }

  auto exprOp = builder.create<ExpressionOp>(
      loc, builder.getIndexType(), ValueRange{key.first, key.second},
      builder.getI64ArrayAttr({1, 1}), builder.getStringAttr("mul"));

  mulCache[key] = exprOp.getResult();
  return exprOp.getResult();
}

/// Decompose a single polynomial constraint
bool decomposePolynomialConstraint(PolynomialConstraintOp pcOp,
                                   ConstraintSpaceOp /*csOp*/) {
  MLIRContext *ctx = pcOp.getContext();
  auto monomials = parseMonomials(pcOp.getMonomials());

  bool hasNonLinear = false;
  for (const auto &m : monomials) {
    if (m.degree() > 1) {
      hasNonLinear = true;
      break;
    }
  }

  if (!hasNonLinear)
    return false;

  LLVM_DEBUG(llvm::dbgs() << "Decomposing polynomial constraint\n");

  OpBuilder builder(pcOp);
  llvm::SmallVector<Value> currentOperands(pcOp.getOperands().begin(),
                                           pcOp.getOperands().end());

  // Cache for created multiplication expressions to avoid redundant ops
  llvm::DenseMap<std::pair<Value, Value>, Value> mulCache;

  // Map value to its index in currentOperands
  llvm::DenseMap<Value, int64_t> valueToIndex;
  for (unsigned i = 0; i < currentOperands.size(); ++i) {
    valueToIndex[currentOperands[i]] = i;
  }

  llvm::SmallVector<Monomial> newMonomials;
  for (auto &m : monomials) {
    if (m.degree() <= 1) {
      newMonomials.push_back(m);
      continue;
    }

    // Decompose degree > 1 term: c * (v1 * v2 * ... * vn)
    Value product = currentOperands[m.varIndices[0]];
    for (size_t i = 1; i < m.varIndices.size(); ++i) {
      Value nextVar = currentOperands[m.varIndices[i]];
      product = getOrMulExpression(builder, pcOp.getLoc(), product, nextVar,
                                   mulCache);

      // Add product to currentOperands if not already there
      if (valueToIndex.find(product) == valueToIndex.end()) {
        valueToIndex[product] = currentOperands.size();
        currentOperands.push_back(product);
      }
    }

    // Now 'product' represents the whole product, it's a degree 1 term in terms
    // of 'product'
    Monomial decomposedM;
    decomposedM.coeff = m.coeff;
    decomposedM.varIndices = {valueToIndex[product]};
    newMonomials.push_back(decomposedM);
  }

  // Create new polynomial constraint op (now technically linear in its
  // operands)
  builder.create<PolynomialConstraintOp>(pcOp.getLoc(), currentOperands,
                                         buildMonomialsAttr(ctx, newMonomials),
                                         pcOp.getUpperBoundAttr());

  pcOp.erase();
  return true;
}

struct LoomConstraintDecompose
    : public impl::LoomConstraintDecomposeBase<LoomConstraintDecompose> {
  using LoomConstraintDecomposeBase::LoomConstraintDecomposeBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](ConstraintSpaceOp csOp) {
      // Collect ops to process
      llvm::SmallVector<PolynomialConstraintOp> pcOps;
      for (auto &op : *csOp.getBodyBlock()) {
        if (auto pcOp = dyn_cast<PolynomialConstraintOp>(&op)) {
          pcOps.push_back(pcOp);
        }
      }

      for (auto pcOp : pcOps) {
        decomposePolynomialConstraint(pcOp, csOp);
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLoomConstraintDecomposePass() {
  return std::make_unique<LoomConstraintDecompose>();
}

} // namespace constraint_opt
} // namespace loom
