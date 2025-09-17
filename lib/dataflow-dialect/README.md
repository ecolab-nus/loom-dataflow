# Dataflow (`df`) Dialect

The custom `df` dialect captures hardware scale-out descriptions that can be paired with affine or Triton-derived kernels. It provides ops to declare spatial dimensions, interconnects between coordinates, and memory operations that take those interconnects as operands.

## Declaring spatial scale-out
- `df.spatial_dim` creates a named spatial dimension with a static extent.
  ```mlir
  %x = df.spatial_dim 8
  %y = df.spatial_dim 8
  ```
  The result value can be used in affine maps and interconnect declarations.

## Describing interconnects
- `df.interconnects` associates an affine map with one or more dimensions to express connectivity. Missing dimensions in the map are implicitly iterated.
  ```mlir
  %horizontal = "df.interconnects"(%x, %y)
      {map = affine_map<(d0, d1) -> (d0 + 1, d1)>}
      : (index, index) -> !df.interconnect
  ```
  In this example each `(x, y)` core has a link to `(x + 1, y)`.

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
