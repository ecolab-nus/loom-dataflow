#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"

#include "DataflowDialect.h.inc"

using namespace mlir;
using namespace tmd::df;

#include "DataflowDialect.cpp.inc"
// Bring in op class declarations for registration below.
#define GET_OP_CLASSES
#include "DataflowOps.h.inc"

void DataflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DataflowOps.cpp.inc"
      >();
}

// Emit op method definitions in this translation unit, like Toy Ch2.
#define GET_OP_CLASSES
#include "DataflowOps.cpp.inc"
