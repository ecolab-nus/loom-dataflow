//===- LoomDialect.td ---------------------------------------*- tablegen -*-===//
//
// LOOM dialect (short name: loom)
//
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"

def Loom_Dialect : Dialect {
  let summary = "LOOM dialect for memory operations and transformations";
  let description = [{
    The `loom` dialect provides operations for memory transformations and
    optimizations, including enhanced reinterpret_cast operations.
  }];

  let name = "loom";
  let cppNamespace = "::loom";
  // Use generated default type printer/parser from types .inc
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = [
    "memref::MemRefDialect"
  ];
}

