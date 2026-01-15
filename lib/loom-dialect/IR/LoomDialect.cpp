//===- LoomDialect.cpp - LOOM Dialect Implementation --------------------===//
//
// Implementation of the LOOM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"

#include "LoomDialect.h.inc"

using namespace mlir;
using namespace loom;

#include "LoomDialect.cpp.inc"
// Bring in op class declarations for registration below.
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

void LoomDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LoomOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "LoomOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstraintSpaceOp Verifier
//===----------------------------------------------------------------------===//

/// Verify that all symbolic variable names within a constraint space are
/// unique.
LogicalResult loom::ConstraintSpaceOp::verify() {
  llvm::DenseMap<StringAttr, Location> variableNames;

  // Walk through all symbolic_var operations in the constraint space body
  for (Operation &op : getBodyBlock()->getOperations()) {
    if (auto symbolicVar = dyn_cast<loom::SymbolicVarOp>(&op)) {
      StringAttr varName = symbolicVar.getNameAttr();

      // Check if this variable name has already been seen
      auto [it, inserted] =
          variableNames.try_emplace(varName, symbolicVar.getLoc());
      if (!inserted) {
        // Duplicate variable name found
        return symbolicVar.emitOpError("duplicate symbolic variable name '")
               << varName.getValue() << "' in constraint space; "
               << "previously defined at " << it->second;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GetSymbolicBlockSizeOp Verifier
//===----------------------------------------------------------------------===//

/// Verify that the symbol reference has the correct format: @space::@var
LogicalResult loom::GetSymbolicBlockSizeOp::verify() {
  SymbolRefAttr symbolRef = getSymbolRef();

  // The symbol reference should have exactly 1 nested reference:
  // - Root: constraint space name (e.g., @global_constraints)
  // - Nested: variable name (e.g., @M)
  if (symbolRef.getNestedReferences().size() != 1) {
    return emitOpError("symbol reference must have format @space::@var, "
                       "got ")
           << symbolRef;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ExpressionOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult loom::ExpressionOp::verify() {
  auto operands = getOperands();
  auto coeffs = getCoeffs();
  auto logic = getLogic();

  if (logic == "add") {
    if (operands.size() != coeffs.size()) {
      return emitOpError(
          "number of operands must match number of coefficients for 'add' "
          "logic");
    }
  } else if (logic == "mul") {
    if (operands.size() != 2) {
      return emitOpError("multiplication must have exactly two operands");
    }
    if (coeffs.size() != 2) {
      return emitOpError(
          "multiplication must have two coefficients (typically {1, 1})");
    }
  } else {
    return emitOpError("unsupported logic type: ") << logic;
  }

  return success();
}
