//===- LoomTypes.td -----------------------------------------*- tablegen -*-===//
// Types for the LOOM dialect (loom)
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "LoomDialect.td"

def SpatialMapType : TypeDef<Loom_Dialect, "SpatialMap"> {
  let mnemonic = "spatial_map";
  let summary = "Symbolic spatial mapping matrix handle type";
  let description = [{
    A handle for a symbolic logical-to-physical spatial mapping matrix.

    The first parameter is the number of logical dimensions (rows), and the
    second parameter is the number of physical mesh dimensions (columns).

    Syntax: `!loom.spatial_map<2x2>`
  }];

  let parameters = (ins "unsigned":$rows, "unsigned":$cols);
  let assemblyFormat = "`<` $rows `x` $cols `>`";
}
