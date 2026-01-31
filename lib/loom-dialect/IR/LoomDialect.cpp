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
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "LoomDialect.h.inc"
#include "llvm/ADT/TypeSwitch.h"

// 1. Declarations
#define GET_TYPEDEF_CLASSES
#include "LoomTypes.h.inc"

#include "LoomEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.h.inc"

#include "LoomDialect.cpp.inc"

#define GET_OP_CLASSES
#include "LoomOps.h.inc"

// 2. Definitions
#define GET_TYPEDEF_CLASSES
#include "LoomTypes.cpp.inc"

#include "LoomEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "LoomAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "LoomOps.cpp.inc"

using namespace mlir;
using namespace loom;

void LoomDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LoomOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "LoomTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "LoomAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ConstraintSpaceOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult loom::ConstraintSpaceOp::verify() {
  llvm::DenseMap<StringAttr, Location> variableNames;
  for (Operation &op : getBodyBlock()->getOperations()) {
    if (auto symbolicVar = dyn_cast<loom::SymbolicVarOp>(&op)) {
      StringAttr varName = symbolicVar.getNameAttr();
      auto [it, inserted] =
          variableNames.try_emplace(varName, symbolicVar.getLoc());
      if (!inserted) {
        return symbolicVar.emitOpError("duplicate symbolic variable name '")
               << varName.getValue() << "' in constraint space; "
               << "previously defined at " << it->second;
      }
    }
  }
  return success();
}

LogicalResult loom::GetSymbolicBlockSizeOp::verify() {
  SymbolRefAttr symbolRef = getSymbolRef();
  if (symbolRef.getNestedReferences().size() != 1) {
    return emitOpError("symbol reference must have format @space::@var, got ")
           << symbolRef;
  }
  return success();
}

LogicalResult loom::ExpressionOp::verify() {
  auto operands = getOperands();
  auto coeffs = getCoeffs();
  auto logic = getLogic();
  if (logic == "add") {
    if (operands.size() != coeffs.size()) {
      return emitOpError("number of operands must match number of coefficients "
                         "for 'add' logic");
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

void loom::CopyToTensorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void loom::CopyFromTensorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}
