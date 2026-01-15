/**
 * @file intermediate_var_compression.cpp
 * @brief Implementation of intermediate variable compression pass.
 */

#include "compress_intermediate_var.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SetVector.h"

#include "LoomDialect.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

#define DEBUG_TYPE "loom-intermediate-var-compression"

namespace loom {
namespace constraint_opt {

using namespace mlir;

namespace {

void compressAddExpression(ExpressionOp exprOp) {
  if (exprOp.getLogic() != "add")
    return;

  Value result = exprOp.getResult();
  auto operands = exprOp.getOperands();
  auto coeffs = exprOp.getCoeffs();

  // Find all LinearConstraintOp that use this result
  llvm::SmallVector<LinearConstraintOp> users;
  for (auto &use : result.getUses()) {
    if (auto lcOp = dyn_cast<LinearConstraintOp>(use.getOwner())) {
      users.push_back(lcOp);
    }
  }

  for (auto lcOp : users) {
    OpBuilder builder(lcOp);
    auto oldOperands = lcOp.getOperands();
    auto oldMap = lcOp.getMap();
    auto isEquality = lcOp.getIsEquality();

    // 1. Determine the new set of operands
    llvm::SmallSetVector<Value, 8> newOperandsSet;
    for (auto v : oldOperands) {
      if (v == result) {
        for (auto op : operands)
          newOperandsSet.insert(op);
      } else {
        newOperandsSet.insert(v);
      }
    }
    auto newOperandsList = newOperandsSet.getArrayRef();

    // 2. Build dimension substitutions
    // Map each dimension of the old map to an expression over the new
    // dimensions.
    llvm::SmallVector<AffineExpr, 8> substitutions;
    for (size_t i = 0; i < oldOperands.size(); ++i) {
      Value v = oldOperands[i];
      if (v == result) {
        // Substitute iv with sum(c_i * op_i)
        AffineExpr expr = builder.getAffineConstantExpr(0);
        for (size_t j = 0; j < operands.size(); ++j) {
          int64_t c = cast<IntegerAttr>(coeffs[j]).getInt();
          Value op = operands[j];
          int newIdx = -1;
          for (size_t k = 0; k < newOperandsList.size(); ++k) {
            if (newOperandsList[k] == op) {
              newIdx = k;
              break;
            }
          }
          assert(newIdx != -1);
          expr = expr + c * builder.getAffineDimExpr(newIdx);
        }
        substitutions.push_back(expr);
      } else {
        // Substitute other variables with their corresponding new dimension
        // index
        int newIdx = -1;
        for (size_t k = 0; k < newOperandsList.size(); ++k) {
          if (newOperandsList[k] == v) {
            newIdx = k;
            break;
          }
        }
        assert(newIdx != -1);
        substitutions.push_back(builder.getAffineDimExpr(newIdx));
      }
    }

    // 3. Create the new AffineMap
    auto newMap = oldMap.replaceDimsAndSymbols(substitutions, {},
                                               newOperandsList.size(), 0);

    // 4. Create new LinearConstraintOp and erase the old one
    builder.create<LinearConstraintOp>(lcOp.getLoc(), newOperandsList,
                                       AffineMapAttr::get(newMap),
                                       builder.getBoolAttr(isEquality));
    lcOp.erase();
  }

  // Finally erase the expression op
  exprOp.erase();
}

} // namespace

LogicalResult runIntermediateVarCompression(ModuleOp module) {
  module.walk([&](ConstraintSpaceOp csOp) {
    // Collect all add expressions first to avoid iterator invalidation
    llvm::SmallVector<ExpressionOp> addExprs;
    for (auto &op : csOp.getBodyBlock()->getOperations()) {
      if (auto exprOp = dyn_cast<ExpressionOp>(&op)) {
        if (exprOp.getLogic() == "add") {
          addExprs.push_back(exprOp);
        }
      }
    }

    // Process them one by one
    // Note: If an add expression depends on another, processing them in order
    // should correctly substitute everything.
    for (auto exprOp : addExprs) {
      compressAddExpression(exprOp);
    }
  });

  return success();
}

} // namespace constraint_opt
} // namespace loom
