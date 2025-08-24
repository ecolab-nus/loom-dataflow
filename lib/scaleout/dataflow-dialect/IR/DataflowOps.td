//===- DataflowOps.td ------------------------------------------*- tablegen -*-===//
// Operations for the TMD Dataflow dialect (df)
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "DataflowTypes.td"

class DF_Op<string mnemonic, list<Trait> traits = []>
  : Op<Dataflow_Dialect, mnemonic, traits> {}

// Bring in hardware declaration ops (df.spatial_dim, df.interconnects)
include "DataflowHwDeclOps.td"

// df.chained_load : similar to affine.load's operands/attributes, plus 'iterator'
def ChainedLoadOp : DF_Op<"chained_load"> {
  let summary = "Load via a chain-like network";
  let description = [{
    `df.chained_load` loads a single element from a memref using affine
    indexing, and additionally carries an iterator selector used to model
    hardware routing along a chain. The `over` clause specifies an SSA handle
    produced by `df.interconnects`.

    Syntax (custom parser/printer):
      `%v = df.chained_load %memref[<affine-exprs-over-ssa-ids>] : memref<...>, over %chain`

    - `%memref` is the memref operand.
    - The affine index list reuses the `affine.load` syntax and semantics, i.e.,
      `multi-dim-affine-map-of-ssa-ids` parsed/printed via
      `parseAffineMapOfSSAIds`/`printAffineMapOfSSAIds`.
    - The result type is the element type of the memref.
    - The trailing clause `, over %chain` sets the chain handle operand.

    Example:
      `%0 = df.chained_load %arg0[%arg1 + 3, %arg2 + 7] : memref<100x100xf32>, over %vertical_chains`
  }];

  let arguments = (ins AnyMemRef:$memref,
                        Variadic<Index>:$indices,
                        AffineMapAttr:$map,
                        InterconnectHandleType:$chain);

  let results = (outs AnyType:$result);

  // Custom parser/printer implements the syntax shown above, including the
  // trailing ", over %chain" clause.
  let hasCustomAssemblyFormat = 1;
}

// df.spatial_wrap: Wrapper op to carry spatial mapping metadata and contain
// transformed code. Printed with a region whose entry block args and
// terminators are elided for readability.
def SpatialWrapOp : DF_Op<"spatial_wrap"> {
  let summary = "Wrapper to annotate spatial mappings over contained region";
  let description = [{
    `df.spatial_wrap` wraps a region of IR (e.g., an affine.parallel) and
    carries an attribute `mapping` which is an array of dictionaries. Each
    dictionary describes one spatial dimension assignment with keys like
    `df_ordinal`, `par_iv`, and `size` (all index-typed integers).

    The custom printer elides entry block arguments and terminators for the
    region body to keep output free of basic block labels and yields.
  }];

  let arguments = (ins ArrayAttr:$mapping);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;
}



