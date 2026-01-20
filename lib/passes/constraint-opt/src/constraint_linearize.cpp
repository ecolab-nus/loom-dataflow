#include "Passes.h"
#include "analysis_engine.h"
#include "constraint_space_utils.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#define DEBUG_TYPE "loom-constraint-linearize"

namespace loom {
namespace constraint_opt {

#define GEN_PASS_DEF_LOOMCONSTRAINTLINEARIZE
#include "Passes.h.inc"

using namespace mlir;
using namespace loom::lcs;

namespace {

/// Emit linear constraints for McCormick relaxation of w = x * y
/// Convention: affine_map expression <= 0
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

  // Lower bound 1: w >= Ly*x + Lx*y - Lx*Ly
  // In <= 0 form: Ly*x + Lx*y - Lx*Ly - w <= 0
  // affine_map<(d0, d1, d2) -> (-d0 + Ly*d1 + Lx*d2 - Lx*Ly)>
  auto map1 = AffineMap::get(3, 0,
                             {(-builder.getAffineDimExpr(0)) +
                              Ly * builder.getAffineDimExpr(1) +
                              Lx * builder.getAffineDimExpr(2) - (Lx * Ly)},
                             ctx);
  builder.create<LinearConstraintOp>(loc, ValueRange{w, x, y},
                                     AffineMapAttr::get(map1));

  // Lower bound 2: w >= Uy*x + Ux*y - Ux*Uy
  // In <= 0 form: Uy*x + Ux*y - Ux*Uy - w <= 0
  auto map2 = AffineMap::get(3, 0,
                             {(-builder.getAffineDimExpr(0)) +
                              Uy * builder.getAffineDimExpr(1) +
                              Ux * builder.getAffineDimExpr(2) - (Ux * Uy)},
                             ctx);
  builder.create<LinearConstraintOp>(loc, ValueRange{w, x, y},
                                     AffineMapAttr::get(map2));

  // Upper bound 1: w <= Uy*x + Lx*y - Lx*Uy
  // In <= 0 form: w - Uy*x - Lx*y + Lx*Uy <= 0
  auto map3 = AffineMap::get(3, 0,
                             {builder.getAffineDimExpr(0) -
                              Uy * builder.getAffineDimExpr(1) -
                              Lx * builder.getAffineDimExpr(2) + (Lx * Uy)},
                             ctx);
  builder.create<LinearConstraintOp>(loc, ValueRange{w, x, y},
                                     AffineMapAttr::get(map3));

  // Upper bound 2: w <= Ly*x + Ux*y - Ux*Ly
  // In <= 0 form: w - Ly*x - Ux*y + Ux*Ly <= 0
  auto map4 = AffineMap::get(3, 0,
                             {builder.getAffineDimExpr(0) -
                              Ly * builder.getAffineDimExpr(1) -
                              Ux * builder.getAffineDimExpr(2) + (Ux * Ly)},
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

  // sum(coeff_i * var_i) <= UB
  // In <= 0 form: sum(coeff_i * var_i) - UB <= 0

  llvm::SmallVector<AffineExpr, 4> exprs;
  AffineExpr sum = builder.getAffineConstantExpr(0);
  for (size_t i = 0; i < monomials.size(); ++i) {
    const auto &m = monomials[i];
    assert(m.varIndices.size() <= 1 &&
           "Polynomial constraint must be decomposed before linearization");

    if (m.varIndices.size() == 1) {
      sum = sum + m.coeff * builder.getAffineDimExpr(m.varIndices[0]);
    } else {
      // Constant term
      sum = sum + m.coeff;
    }
  }

  // sum - UB <= 0
  AffineExpr resultExpr = sum - ub;
  auto map = AffineMap::get(operands.size(), 0, {resultExpr}, ctx);

  builder.create<LinearConstraintOp>(pcOp.getLoc(), operands,
                                     AffineMapAttr::get(map));
  pcOp.erase();
}

struct LoomConstraintLinearize
    : public impl::LoomConstraintLinearizeBase<LoomConstraintLinearize> {
  using LoomConstraintLinearizeBase::LoomConstraintLinearizeBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](ConstraintSpaceOp csOp) {
      BoundInferenceService bis(csOp);
      bis.initialize();

      OpBuilder builder(csOp.getContext());

      // 1. Process polynomial_constraints (which should now be formally linear
      // or decomposed)
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

        if (exprOp.getLogic() == "add") {
          continue;
        }

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
        }

        // Replace original result with the new local variable
        exprOp.getResult().replaceAllUsesWith(ivOp.getResult());
        exprOp.erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLoomConstraintLinearizePass() {
  return std::make_unique<LoomConstraintLinearize>();
}

} // namespace constraint_opt
} // namespace loom
