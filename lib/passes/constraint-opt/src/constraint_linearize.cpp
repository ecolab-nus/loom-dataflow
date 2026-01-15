/**
 * @file constraint_linearize.cpp
 * @brief Implementation of polynomial constraint linearization pass.
 */

#include "constraint_linearize.h"
#include "analysis_engine.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#define DEBUG_TYPE "loom-constraint-linearize"

namespace loom {
namespace constraint_opt {

using namespace mlir;
using namespace loom::lcs;

namespace {

struct ParsedMonomial {
  int64_t coeff;
  llvm::SmallVector<int64_t, 4> vars;
};

llvm::SmallVector<ParsedMonomial> parseMonomials(ArrayAttr attr) {
  llvm::SmallVector<ParsedMonomial> result;
  if (!attr)
    return result;
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

/// Emit linear constraints for McCormick relaxation of w = x * y
void emitMcCormickConstraints(OpBuilder &builder, Location loc, Value w,
                              Value x, Value y, BoundInferenceService &bis) {
  Interval ix = bis.getRange(x);
  Interval iy = bis.getRange(y);

  if (!ix.isValid() || !iy.isValid()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Warning: Invalid bounds for McCormick relaxation\n");
    return;
  }

  int64_t Lx = ix.lower;
  int64_t Ux = ix.upper;
  int64_t Ly = iy.lower;
  int64_t Uy = iy.upper;

  MLIRContext *ctx = builder.getContext();

  // 1. w - Ly*x - Lx*y + Lx*Ly >= 0
  // affine_map<(d0, d1, d2) -> (d0 - Ly*d1 - Lx*d2 + Lx*Ly)>
  auto map1 = AffineMap::get(3, 0,
                             {builder.getAffineDimExpr(0) -
                              Ly * builder.getAffineDimExpr(1) -
                              Lx * builder.getAffineDimExpr(2) + (Lx * Ly)},
                             ctx);
  builder.create<LinearConstraintOp>(loc, ValueRange{w, x, y},
                                     AffineMapAttr::get(map1));

  // 2. w - Uy*x - Ux*y + Ux*Uy >= 0
  auto map2 = AffineMap::get(3, 0,
                             {builder.getAffineDimExpr(0) -
                              Uy * builder.getAffineDimExpr(1) -
                              Ux * builder.getAffineDimExpr(2) + (Ux * Uy)},
                             ctx);
  builder.create<LinearConstraintOp>(loc, ValueRange{w, x, y},
                                     AffineMapAttr::get(map2));

  // 3. -w + Uy*x + Lx*y - Lx*Uy >= 0
  auto map3 = AffineMap::get(3, 0,
                             {(-builder.getAffineDimExpr(0)) +
                              Uy * builder.getAffineDimExpr(1) +
                              Lx * builder.getAffineDimExpr(2) - (Lx * Uy)},
                             ctx);
  builder.create<LinearConstraintOp>(loc, ValueRange{w, x, y},
                                     AffineMapAttr::get(map3));

  // 4. -w + Ly*x + Ux*y - Ux*Ly >= 0
  auto map4 = AffineMap::get(3, 0,
                             {(-builder.getAffineDimExpr(0)) +
                              Ly * builder.getAffineDimExpr(1) +
                              Ux * builder.getAffineDimExpr(2) - (Ux * Ly)},
                             ctx);
  builder.create<LinearConstraintOp>(loc, ValueRange{w, x, y},
                                     AffineMapAttr::get(map4));
}

void linearizePolynomialConstraint(PolynomialConstraintOp pcOp,
                                   OpBuilder &builder) {
  MLIRContext *ctx = pcOp.getContext();
  auto monomials = parseMonomials(pcOp.getMonomials());
  auto operands = pcOp.getOperands();
  int64_t ub = pcOp.getUpperBound();

  // sum(coeff_i * var_i) <= UB  =>  -sum(coeff_i * var_i) + UB >= 0

  llvm::SmallVector<AffineExpr, 4> exprs;
  AffineExpr sum = builder.getAffineConstantExpr(0);
  for (size_t i = 0; i < monomials.size(); ++i) {
    const auto &m = monomials[i];
    assert(m.vars.size() <= 1 &&
           "Polynomial constraint must be decomposed before linearization");

    if (m.vars.size() == 1) {
      sum = sum + m.coeff * builder.getAffineDimExpr(m.vars[0]);
    } else {
      // Constant term
      sum = sum + m.coeff;
    }
  }

  // -sum + UB >= 0
  AffineExpr resultExpr = (-sum) + ub;
  auto map = AffineMap::get(operands.size(), 0, {resultExpr}, ctx);

  builder.create<LinearConstraintOp>(pcOp.getLoc(), operands,
                                     AffineMapAttr::get(map));
  pcOp.erase();
}

} // namespace

LogicalResult runConstraintLinearize(ModuleOp module) {
  module.walk([&](ConstraintSpaceOp csOp) {
    BoundInferenceService bis(csOp);
    bis.initialize();

    OpBuilder builder(csOp.getContext());

    // 1. Process polynomial_constraints (which should now be formally linear or
    // decomposed)
    llvm::SmallVector<PolynomialConstraintOp> pcOps;
    for (auto &op : csOp.getBodyBlock()->getOperations()) {
      if (auto pcOp = dyn_cast<PolynomialConstraintOp>(&op)) {
        pcOps.push_back(pcOp);
      }
    }

    for (auto pcOp : pcOps) {
      builder.setInsertionPoint(pcOp);
      linearizePolynomialConstraint(pcOp, builder);
    }

    // 2. Process all ExpressionOps in correct order (program order)
    // We collect them first because we will be erasing them.
    llvm::SmallVector<ExpressionOp> exprOps;
    for (auto &op : csOp.getBodyBlock()->getOperations()) {
      if (auto exprOp = dyn_cast<ExpressionOp>(&op)) {
        exprOps.push_back(exprOp);
      }
    }

    for (auto exprOp : exprOps) {
      Location loc = exprOp.getLoc();
      builder.setInsertionPoint(exprOp);

      // Create a new intermediate variable to replace this expression
      auto ivOp =
          builder.create<IntermediateVarOp>(loc, builder.getIndexType());

      // Propagate range information to the new variable
      bis.setRange(ivOp.getResult(), bis.getRange(exprOp.getResult()));

      if (exprOp.getLogic() == "mul") {
        // Emit McCormick constraints for multiplication
        auto operands = exprOp.getOperands();
        emitMcCormickConstraints(builder, loc, ivOp.getResult(), operands[0],
                                 operands[1], bis);
      } else if (exprOp.getLogic() == "add") {
        // Emit equality constraint for addition: iv = sum(coeffs * operands)
        auto operands = exprOp.getOperands();
        auto coeffs = exprOp.getCoeffs();

        llvm::SmallVector<Value> linearOperands;
        linearOperands.push_back(ivOp.getResult());
        for (auto v : operands)
          linearOperands.push_back(v);

        // Map: (iv, op1, op2, ...) -> iv - sum(coeff_i * op_i)
        AffineExpr expr = builder.getAffineDimExpr(0);
        for (unsigned i = 0; i < operands.size(); ++i) {
          int64_t c = cast<IntegerAttr>(coeffs[i]).getInt();
          expr = expr - c * builder.getAffineDimExpr(i + 1);
        }

        auto map = AffineMap::get(linearOperands.size(), 0, {expr},
                                  builder.getContext());
        builder.create<LinearConstraintOp>(loc, linearOperands,
                                           AffineMapAttr::get(map),
                                           builder.getBoolAttr(true));
      }

      // Replace original result with the new local variable
      exprOp.getResult().replaceAllUsesWith(ivOp.getResult());
      exprOp.erase();
    }
  });

  return success();
}

} // namespace constraint_opt
} // namespace loom
