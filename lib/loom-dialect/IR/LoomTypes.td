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
  let summary = "(DEPRECATED) Physical buffer placeholder - use memref instead";
  let description = [{
    DEPRECATED: This type is being phased out in favor of standard memref.
    The `loom.alloc` operation now returns memref types directly.
    
    An opaque type representing a physical memory allocation record in L1.
    
    Syntax: `!loom.buffer_token`
  }];
}
