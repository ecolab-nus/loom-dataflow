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
    `df.chained_load` models a hardware data-movement: loading an element from
    a memref using affine indexing, routed through a chain along one hardware
    iterator dimension. The `iterator` attribute must correspond to one of the
    iterators of the innermost enclosing `affine.parallel`.

    Syntax (custom printer to match affine-like style):
      `%v = df.chained_load %memref[<affine-exprs>] iterator(<piv>) : memref<T>`
  }];

  let arguments = (ins AnyMemRef:$memref,
                        Variadic<Index>:$indices,
                        AffineMapAttr:$map,
                        I64Attr:$iterator);

  let results = (outs AnyType:$result);

  // Declarative assembly format
  let assemblyFormat = [{
    $memref `[` $indices `]`
    `iterator` `(` $iterator `)`
    `map` `(` $map `)` attr-dict `:` type($memref) `,` type($result)
  }];
}



