//===- DataflowTypes.td -------------------------------------*- tablegen -*-===//
// Types for the TMD Dataflow dialect (df)
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "DataflowDialect.td"

// A minimal, opaque handle type representing an interconnect configuration
// Printed as: !df.interconnect
def InterconnectHandleType : TypeDef<Dataflow_Dialect, "InterconnectHandle"> {
  let mnemonic = "interconnect";
  let summary = "Opaque interconnect handle type";
  let description = [{
    An opaque handle representing an interconnect configuration declared
    by `df.interconnects`. This type carries no semantics by itself and exists to
    avoid reusing unrelated builtin types (e.g., `i64`).

    Syntax: `!df.interconnect`
  }];
}

 