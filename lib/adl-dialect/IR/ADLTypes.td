//===- ADLTypes.td ---------------------------------------------*- tablegen -*-===//
// Types for the ADL dialect
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "ADLDialect.td"

def ArchHandleType : TypeDef<ADL_Dialect, "ArchHandle"> {
  let mnemonic = "arch";
  let summary = "Opaque architecture handle type";
  let description = [{
    An opaque handle representing an architectural component: a processor,
    a composed architecture, or a scaled architecture.

    Syntax: `!adl.arch`
  }];
  // Needed so declarative assemblyFormat can reconstruct the type without
  // an explicit `: !adl.arch` annotation in the IR.
  let builderCall = "$_builder.getType<::adl::ArchHandleType>()";
}

def MemHandleType : TypeDef<ADL_Dialect, "MemHandle"> {
  let mnemonic = "mem";
  let summary = "Opaque memory handle type";
  let description = [{
    An opaque handle representing a memory resource: a memory bank or
    a memory array.

    Syntax: `!adl.mem`
  }];
  // Needed so declarative assemblyFormat can reconstruct the type without
  // an explicit `: !adl.mem` annotation in the IR.
  let builderCall = "$_builder.getType<::adl::MemHandleType>()";
}
