//===- LoomTypes.td -----------------------------------------*- tablegen -*-===//
// Types for the LOOM dialect (loom)
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "LoomDialect.td"

def ViewType : TypeDef<Loom_Dialect, "View"> {
  let mnemonic = "view";
  let summary = "(DEPRECATED) Logical view type - use memref instead";
  let description = [{
    DEPRECATED: This type is no longer used by `loom.view`. 
    The `loom.view` operation now returns standard `memref` types.
    This type is kept only for backward compatibility with older MLIR files.
    
    Syntax: `!loom.view`
  }];
}


def BufferTokenType : TypeDef<Loom_Dialect, "BufferToken"> {
  let mnemonic = "buffer_token";
  let summary = "Physical buffer placeholder token for L1 memory";
  let description = [{
    An opaque type representing a physical memory allocation record in L1.
    This type serves as a placeholder for L1 memory allocation and is used
    to track buffer ownership in the SSA chain.
    
    Syntax: `!loom.buffer_token`
  }];
}
