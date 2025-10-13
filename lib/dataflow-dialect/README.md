# Dataflow (`df`) Dialect

The custom `df` dialect captures hardware scale-out descriptions that can be paired with affine or Triton-derived kernels. It provides ops to declare spatial dimensions, interconnects between coordinates, and memory operations that take those interconnects as operands.

## Declaring spatial scale-out
- `df.spatial_dim` creates a named spatial dimension with a static extent.
  ```mlir
  %x = df.spatial_dim "x", 8
  %y = df.spatial_dim "y", 8
  ```
  The result value can be used in affine maps and interconnect declarations.

## Declaring compute and memory fabrics
- `df.compute` derives a compute fabric from spatial dimensions. The affine map captures how indices tile the hardware.
  ```mlir
  %cores = "df.compute"(%x, %y)
      {label = "compute", map = affine_map<(d0, d1) -> (d0, d1)>}
      : (index, index) -> !df.compute
  ```
- `df.memory` describes memory resources in a similar fashion.
  ```mlir
  %mems = "df.memory"(%x)
      {label = "memory", map = affine_map<(d0) -> (d0)>}
      : (index) -> !df.memory
  ```

## Routing across the fabric
- `df.interconnects` connects two compute or memory handle sets. The operands identify the source and destination resources while the affine map describes how indices are transformed between them. A user-facing string label appears first in the custom assembly.
  ```mlir
  %horizontal = df.interconnects "horizontal_links" %cores : !df.compute, %cores : !df.compute, %x, %y
      {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !df.interconnect
  ```
  Here each compute tile `(x, y)` links to its neighbor `(x + 1, y)`.
- `df.mux` relates compute and memory handles, defining how compute tiles share resources.
  ```mlir
  %core_to_mem = "df.mux"(%cores, %mems, %x, %y)
      {map = affine_map<(d0, d1) -> (d0)>}
      : (!df.compute, !df.memory, index, index) -> !df.mux
  ```
  Omitting result dimensions implies sharing; here all cores with the same `x` coordinate use the same memory tile.

## Memory operations aware of topology
- `df.chained_load` models a memory load that traverses a chain-like interconnect.
  ```mlir
  %val = df.chained_load %buffer[%i + 3, %j + 7]
           : memref<100x100xf32>, over %vertical
  ```
  The `over` operand ties the operation to a previously declared interconnect so passes can reason about movement costs.

## Dialect implementation
- TableGen files (`DataflowDialect.td`, `DataflowOps.td`, `DataflowTypes.td`) live beside the C++ dialect implementation (`DataflowDialect.cpp`).
- The generated headers are exposed through the `tmdDataflowDialect` library, which is linked by both the passes and the command-line tools.

Additional design notes and examples live in `test/Dialect/DataflowDialect` and the higher-level READMEs under `lib/` and `tool/`.
