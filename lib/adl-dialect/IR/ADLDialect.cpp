#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ADLDialect.h.inc"
// Generated type declarations
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.h.inc"
// Generated type definitions (TypeID, printers/parsers)
#define GET_TYPEDEF_CLASSES
#include "ADLTypes.cpp.inc"

using namespace mlir;
using namespace adl;

#include "ADLDialect.cpp.inc"
// Bring in op class declarations for registration below.
#define GET_OP_CLASSES
#include "ADLOps.h.inc"

void ADLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ADLOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "ADLTypes.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "ADLOps.cpp.inc"
