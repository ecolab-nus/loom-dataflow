// Minimal op glue; no custom verification or builders needed.
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

#define GET_OP_CLASSES
#include "DataflowOps.h.inc"
#define GET_OP_CLASSES
#include "DataflowOps.cpp.inc"
