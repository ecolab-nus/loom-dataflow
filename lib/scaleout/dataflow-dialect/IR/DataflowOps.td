//===- DataflowOps.td ------------------------------------------*- tablegen -*-===//
// Operations for the TMD Dataflow dialect (df)
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "DataflowDialect.td"

class DF_Op<string mnemonic, list<Trait> traits = []>
  : Op<Dataflow_Dialect, mnemonic, traits> {}

// df.chained_load : similar to affine.load's operands/attributes, plus 'iterator'
def ChainedLoadOp : DF_Op<"chained_load"> {
  let summary = "Load via a chain-like network";
  let description = [{
    `df.chained_load` loads a single element from a memref using affine
    indexing, and additionally carries an iterator selector used to model
    hardware routing along a chain. The `iterator` value is a 64-bit integer
    and should correspond to a valid hardware iterator dimension (e.g., one of
    the iterators of the innermost enclosing `affine.parallel`).

    Syntax (custom parser/printer):
      `%v = df.chained_load %memref[<affine-exprs-over-ssa-ids>] : memref<...>, over iter(<VI>)`

    - `%memref` is the memref operand.
    - The affine index list reuses the `affine.load` syntax and semantics, i.e.,
      `multi-dim-affine-map-of-ssa-ids` parsed/printed via
      `parseAffineMapOfSSAIds`/`printAffineMapOfSSAIds`.
    - The result type is the element type of the memref.
    - The trailing clause `, over iter(VI)` sets the integral `iterator`.

    Example:
      `%0 = df.chained_load %arg0[%arg1 + 3, %arg2 + 7] : memref<100x100xf32>, over iter(5)`
  }];

  let arguments = (ins AnyMemRef:$memref,
                        Variadic<Index>:$indices,
                        AffineMapAttr:$map,
                        I64Attr:$iterator);

  let results = (outs AnyType:$result);

  // Custom parser/printer implements the syntax shown above, including the
  // trailing ", over iter(VI)" clause.
  let hasCustomAssemblyFormat = 1;
}



