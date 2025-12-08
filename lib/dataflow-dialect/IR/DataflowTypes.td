//===- DataflowTypes.td -------------------------------------*- tablegen -*-===//
// Types for the LOOM Dataflow dialect (df)
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

def ComputeHandleType : TypeDef<Dataflow_Dialect, "ComputeHandle"> {
  let mnemonic = "compute";
  let summary = "Opaque compute tile handle type";
  let description = [{
    An opaque handle representing a set of compute tiles declared by `df.compute`.
    The handle can be passed to other operations, such as `df.mux`, to describe
    relationships involving compute resources.

    Syntax: `!df.compute`
  }];
}

def MemoryHandleType : TypeDef<Dataflow_Dialect, "MemoryHandle"> {
  let mnemonic = "memory";
  let summary = "Opaque memory tile handle type";
  let description = [{
    An opaque handle representing a set of memory resources declared by
    `df.memory`. The handle can be used to model data placement within the
    spatial fabric.

    Syntax: `!df.memory`
  }];
}

def MuxHandleType : TypeDef<Dataflow_Dialect, "MuxHandle"> {
  let mnemonic = "mux";
  let summary = "Opaque mux connectivity handle type";
  let description = [{
    An opaque handle representing a multiplexing relationship between compute
    and memory resources declared via `df.mux`. The handle can be referenced by
    downstream passes to reason about how compute tiles share memories or other
    control endpoints.

    Syntax: `!df.mux`
  }];
}

def FunctionalUnitHandleType : TypeDef<Dataflow_Dialect, "FunctionalUnitHandle"> {
  let mnemonic = "functional_unit";
  let summary = "Opaque functional unit handle type";
  let description = [{
    An opaque handle representing a functional unit (e.g., matrix unit, vector unit)
    declared by `df.mat` or `df.vec`. The handle can be referenced in core
    declarations to describe the compute components available within each core.

    Syntax: `!df.functional_unit`
  }];
}

 