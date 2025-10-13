//===- DataflowHwDeclOps.td ----------------------------------*- tablegen -*-===//
// Hardware declaration operations for the TMD Dataflow dialect (df)
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"

// NOTE: This file is intended to be included from `DataflowOps.td`, which
// defines the helper `DF_Op` class and sets up codegen for the df dialect ops.

def HandleLike : TypeConstraint<CPred<
  "::llvm::isa<::tmd::df::ComputeHandleType>($_self) || " #
  "::llvm::isa<::tmd::df::MemoryHandleType>($_self)">>;

// df.spatial_dim : declare a spatial dimension with a name and a given size
// Example:
//   %x = df.spatial_dim "x", 8
// The result is an index-typed SSA value that can participate in affine maps.
def SpatialDimOp : DF_Op<"spatial_dim"> {
  let summary = "Declare a spatial dimension with a name and a fixed size";
  let description = [{
    `df.spatial_dim` declares a hardware spatial dimension with a user-facing
    name and a constant size. The result is an index-typed SSA value intended
    to be referenced in affine expressions (e.g., in `df.interconnects`).

    Example:
      `%x = df.spatial_dim "x", 8`
  }];

  let arguments = (ins StrAttr:$name, I64Attr:$size);
  let results = (outs Index:$dim);

  let assemblyFormat = "$name `,` $size attr-dict";
}

// df.interconnects : declare chain interconnects between handles using an affine
// expression over declared spatial dimensions. This produces an opaque handle
// representing the chain configuration.
// Example (matches current custom assembly):
//   %cores_to_cores = df.interconnects "horizontal_links" %cores : !df.compute, %cores: !df.compute, %x, %y
//       {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !df.interconnect
//   %mems_to_mems = df.interconnects "ring" %memories : !df.memory, %memories: !df.memory, %x
//       {map = affine_map<(d0) -> (d0 + 1)>} : !df.interconnect
def InterconnectsOp : DF_Op<"interconnects"> {
  let summary = "Declare interconnects via an affine expression";
  let description = [{
    `df.interconnects` declares chain-like interconnects among the spatial
    dimensions using an affine expression over SSA ids (typically produced by
    `df.spatial_dim`). The operation requires explicit compute or memory
    handles describing the entities that participate in the interconnect. The
    affine expression is provided via the `map` attribute (same textual form as
    other affine maps) alongside the SSA operands used to materialize the
    expression.

    The operation also carries an optional user-facing string label that is
    printed first in the custom assembly (e.g., `"horizontal_links"`).

    Examples (compute-to-compute and memory-to-memory):
      `%cores_to_cores = df.interconnects "horizontal_links" %cores : !df.compute, %cores: !df.compute, %x, %y
          {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !df.interconnect`
      `%mems_to_mems = df.interconnects "ring" %memories : !df.memory, %memories: !df.memory, %x
          {map = affine_map<(d0) -> (d0 + 1)>} : !df.interconnect`
  }];

  let arguments = (ins StrAttr:$label,
                       HandleLike:$source,
                       HandleLike:$target,
                       Variadic<Index>:$indices,
                       AffineMapAttr:$map);
  let results = (outs InterconnectHandleType:$handle);

  let assemblyFormat = "$label $source `:` type($source) `,` $target `:` type($target) `,` $indices attr-dict `:` type($handle)";
}

// df.compute : declare compute resources derived from spatial dimensions.
// Example (matches current custom assembly):
//   %cores = df.compute "cluster_cores", %x, %y
//       {map = affine_map<(d0, d1) -> (d0, d1)>}
def ComputeOp : DF_Op<"compute"> {
  let summary = "Declare compute resources via an affine map";
  let description = [{
    `df.compute` declares compute tiles derived from previously defined spatial
    dimensions. The operation records an affine map that specifies how the
    spatial dimensions index into the compute fabric.

    The result is an opaque `!df.compute` handle that can be consumed by other
    operations such as `df.mux` when describing connectivity.

    Example:
      `%cores = df.compute "cluster_cores", %x, %y
          {map = affine_map<(d0, d1) -> (d0, d1)>}`
  }];

  let arguments = (ins StrAttr:$label,
                       Variadic<Index>:$indices,
                       AffineMapAttr:$map);
  let results = (outs ComputeHandleType:$handle);

  let assemblyFormat = "$label `,` $indices attr-dict";
}

// df.memory : declare memory resources derived from spatial dimensions.
// Example (matches current custom assembly):
//   %memories = df.memory "cluster_mem", %x
//       {map = affine_map<(d0) -> (d0)>}
def MemoryOp : DF_Op<"memory"> {
  let summary = "Declare memory resources via an affine map";
  let description = [{
    `df.memory` declares memory tiles associated with spatial dimensions and
    records their layout via an affine map. The resulting `!df.memory` handle
    is an opaque reference to those resources.

    Example:
      `%memories = df.memory "cluster_mem", %x
          {map = affine_map<(d0) -> (d0)>}`
  }];

  let arguments = (ins StrAttr:$label,
                       Variadic<Index>:$indices,
                       AffineMapAttr:$map);
  let results = (outs MemoryHandleType:$handle);

  let assemblyFormat = "$label `,` $indices attr-dict";
}

// df.mux : declare multiplexing between computes and memories (or other
// resources) using an affine expression over compute indices.
// Example (matches current custom assembly):
//   %core_to_mem = df.mux %cores: !df.compute, %memories: !df.memory, %x, %y
//       {map = affine_map<(d0, d1) -> (d0)>}
def MuxOp : DF_Op<"mux"> {
  let summary = "Declare multiplexing between compute and memory resources";
  let description = [{
    `df.mux` expresses how compute resources share memory or control
    endpoints. The affine map consumes indices associated with the compute
    fabric and produces coordinates into the target resource space. Missing
    target dimensions imply sharing (multiplexing) across the omitted axes.

    The operation returns an opaque `!df.mux` handle for downstream analyses.

    Example:
      `%core_to_mem = df.mux %cores: !df.compute, %memories: !df.memory, %x, %y
          {map = affine_map<(d0, d1) -> (d0)>}`
  }];

  let arguments = (ins HandleLike:$source,
                       HandleLike:$target,
                       Variadic<Index>:$indices,
                       AffineMapAttr:$map);
  let results = (outs MuxHandleType:$handle);

  let assemblyFormat = "$source `:` type($source) `,` $target `:` type($target) `,` $indices attr-dict";
}


