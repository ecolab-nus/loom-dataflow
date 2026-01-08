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

/// Verify that all symbolic variable names within a constraint space are unique.
LogicalResult loom::ConstraintSpaceOp::verify() {
  llvm::DenseMap<StringAttr, Location> variableNames;
  
  // Walk through all symbolic_var operations in the constraint space body
  for (Operation &op : getBodyBlock()->getOperations()) {
    if (auto symbolicVar = dyn_cast<loom::SymbolicVarOp>(&op)) {
      StringAttr varName = symbolicVar.getNameAttr();
      
      // Check if this variable name has already been seen
      auto [it, inserted] = variableNames.try_emplace(varName, symbolicVar.getLoc());
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

