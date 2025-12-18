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

