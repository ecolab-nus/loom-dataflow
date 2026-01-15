/**
 * @file constraint_decompose.cpp
 * @brief Implementation of polynomial constraint decomposition pass.
 */

#include "constraint_decompose.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#include <algorithm>
#include <map>
#include <vector>

#define DEBUG_TYPE "loom-constraint-decompose"

namespace loom {
namespace constraint_opt {

using namespace mlir;

namespace {

/// Parsed monomial for decomposition
struct ParsedMonomial {
  int64_t coeff;
  llvm::SmallVector<int64_t, 4> vars;

  size_t degree() const { return vars.size(); }
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
                                   ConstraintSpaceOp csOp) {
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

  llvm::SmallVector<ParsedMonomial> newMonomials;
  for (auto &m : monomials) {
    if (m.degree() <= 1) {
      newMonomials.push_back(m);
      continue;
    }

    // Decompose degree > 1 term: c * (v1 * v2 * ... * vn)
    Value product = currentOperands[m.vars[0]];
    for (size_t i = 1; i < m.vars.size(); ++i) {
      Value nextVar = currentOperands[m.vars[i]];
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
    ParsedMonomial decomposedM;
    decomposedM.coeff = m.coeff;
    decomposedM.vars = {valueToIndex[product]};
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

} // namespace

LogicalResult runConstraintDecompose(ModuleOp module) {
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

  return success();
}

} // namespace constraint_opt
} // namespace loom
