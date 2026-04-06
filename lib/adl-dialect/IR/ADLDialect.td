//===- ADLDialect.td -------------------------------------------*- tablegen -*-===//
//
// ADL (Architecture Description Language) dialect
//
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"

def ADL_Dialect : Dialect {
  let summary = "Architecture Description Language dialect";
  let description = [{
    The `adl` dialect describes hardware architecture topology: spatial
    dimensions, memory banks/arrays, processors, and composition/scaling
    of architectural building blocks.
  }];

  let name = "adl";
  let cppNamespace = "::adl";
  let useDefaultTypePrinterParser = 1;
}
