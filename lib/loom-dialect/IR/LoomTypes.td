//===- LoomTypes.td -----------------------------------------*- tablegen -*-===//
// Types for the LOOM dialect (loom)
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "LoomDialect.td"

def ViewType : TypeDef<Loom_Dialect, "View"> {
  let mnemonic = "view";
  let summary = "Logical view type encapsulating coordinate metadata";
  let description = [{
    An opaque type representing a logical view (coordinate package).
    It encapsulates a reference to a source memref along with offsets, 
    sizes, and strides for defining a slice.
    
    This type acts as SSA chain glue without producing any runtime 
    instructions. It represents a "logical perspective with coordinate 
    information" in the SSA graph.
    
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
