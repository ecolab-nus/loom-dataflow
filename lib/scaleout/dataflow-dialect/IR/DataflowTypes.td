//===- DataflowTypes.td -------------------------------------*- tablegen -*-===//
// Types for the TMD Dataflow dialect (df)
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "DataflowDialect.td"

// A minimal, opaque handle type representing a chain interconnect
// configuration. Printed as: !df.chain
def ChainHandleType : TypeDef<Dataflow_Dialect, "ChainHandle"> {
  let mnemonic = "chain";
  let summary = "Opaque chain handle type";
  let description = [{
    An opaque handle representing a chain interconnect configuration declared
    by `df.chains`. This type carries no semantics by itself and exists to
    avoid reusing unrelated builtin types (e.g., `i64`).

    Syntax: `!df.chain`
  }];
}

 