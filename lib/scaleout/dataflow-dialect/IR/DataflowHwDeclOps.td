//===- DataflowHwDeclOps.td ----------------------------------*- tablegen -*-===//
// Hardware declaration operations for the TMD Dataflow dialect (df)
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"

// NOTE: This file is intended to be included from `DataflowOps.td`, which
// defines the helper `DF_Op` class and sets up codegen for the df dialect ops.

// df.spatial_dim : declare a spatial dimension of a given static size
// Example:
//   %x = df.spatial_dim 8
// The result is an index-typed SSA value that can participate in affine maps.
def SpatialDimOp : DF_Op<"spatial_dim"> {
  let summary = "Declare a spatial dimension with a fixed size";
  let description = [{
    `df.spatial_dim` declares a hardware spatial dimension with a constant
    size. The result is an index-typed SSA value intended to be referenced in
    affine expressions (e.g., in `df.chains`).

    Example:
      `%x = df.spatial_dim 8`
  }];

  let arguments = (ins I64Attr:$size);
  let results = (outs Index:$dim);

  // Simple custom-looking format without explicit types.
  let assemblyFormat = "$size attr-dict";
}

// df.chains : declare chain interconnects using an affine expression over
// declared spatial dimensions. This produces an opaque handle (i64 for now)
// representing the chain configuration.
// Example (conceptual syntax per README):
//   %horiz = df.chains[%x, %x + 1]
def ChainsOp : DF_Op<"chains"> {
  let summary = "Declare chain interconnects via an affine expression";
  let description = [{
    `df.chains` declares chain-like interconnects among the spatial
    dimensions using an affine expression over SSA ids (typically produced by
    `df.spatial_dim`). The affine list reuses the `affine.load` syntax and is
    stored in the `map` attribute alongside the SSA operands used to
    materialize the expression.

    Example:
      `%horizontal = df.chains[%x, %x + 1]`

    Note: This operation currently returns an `i64` handle as a placeholder.
    A dedicated type may be introduced later.
  }];

  let arguments = (ins Variadic<Index>:$indices, AffineMapAttr:$map);
  let results = (outs I64:$handle);

  // Use generic-like printing to avoid a custom parser implementation here.
  // Users will see/accept the generic form unless/ until a custom
  // parser/printer is added in C++.
}


