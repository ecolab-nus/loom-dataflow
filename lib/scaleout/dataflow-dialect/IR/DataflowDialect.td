//===- DataflowDialect.td ---------------------------------------*- tablegen -*-===//
//
// TMD Dataflow dialect (short name: df)
//
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"

def Dataflow_Dialect : Dialect {
  let summary = "Dataflow dialect for hardware-aware data movement";
  let description = [{
    The `df` dialect models hardware-aware data movement and collective
    communication patterns. It starts with a `df.chained_load` operation
    that represents loading data via a chain-like network.
  }];

  let name = "df";
  let cppNamespace = "::tmd::df";

  let dependentDialects = [
    "affine::AffineDialect",
    "memref::MemRefDialect"
  ];
}



