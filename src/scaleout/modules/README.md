## Modules
Resources are the primitive building blocks (SRAM capacity/ports, rings, chains). **Modules** are higher-level compositions of these resources that represent concrete behaviors for using resources. A module encapsulates:

- **what resources it requires**
- **how it consumes/releases those resources** when an operation is mapped

Module instances do not own or list the cores. Core linkage and placement are modeled by a specialized `NetworkModule` (see below), which keeps an affine description of the linked cores. The affine map is mandatory.

Compiler integration (declaring which IR ops can map to a module and with which constraints) is planned via PDLL generation, but is not implemented yet.

## Provided primitive modules

### Torus (ring-backed)
Uses a `Ring` interconnect resource for broadcast/reduction collectives. Requires a `NetworkModule` to describe which cores are linked and their affine placement.

- Broadcast usage example:
  - Pseudocode
    ```
    parallel_for x
      LOAD A[nX]
    ```
  - `x` is affine-compatible with the cores on the torus. `nX` does not depend on `x`.
  - Mapping consumes the `Ring` and, if modeled, deducts one element of `A` from the local `MemoryCapacity` of each participating core.

- Reduction usage example:
  - Pseudocode
    ```
    reduction_for x
      REDUCTION_OP(x)
    ```
  - Mapping consumes the `Ring`. Per-core memory is not affected by default.

### 2D Mesh
A 2D mesh of size (N, M) is modeled as N horizontal `Chain`s (left–right) and M vertical `Chain`s (top–down). Requires a `NetworkModule` to describe which cores participate and their affine placement.

- Each row/column behaves like a chain (no reduction support).
- Use Torus for reduction-optimized collectives.

## C++ API (initial)
This directory defines minimal C++ classes to model modules:

- `Module`: base base class holding only a name; it does not link cores.
- `NetworkModule`: derived class that links cores and keeps affine placement via a mandatory MLIR `AffineMap`.
- `Torus`: requires a `NetworkModule`; wraps a `Ring` and optional per-core `MemoryCapacity` for broadcast accounting. Methods:
  - `canBroadcast(element_bytes)`
  - `acquireForBroadcast(element_bytes)` / `releaseBroadcast(element_bytes)`
  - `acquireForReduction()` / `releaseReduction()`
- `Mesh2D`: requires a `NetworkModule`; wraps horizontal and vertical `Chain`s. Methods:
  - `isAvailable()`
  - `acquire()` / `release()`

## NetworkModule
`NetworkModule` represents the linkage of cores and their affine placement. It stores:

- a list of core ids
- a mandatory MLIR `AffineMap` describing the mapping from logical coordinates to core ids. Example: `(i, j)[N] -> (i * N + j)`

Use it alongside concrete modules (e.g., `Torus`, `Mesh2D`).

Compiler/IR matching hooks are intentionally omitted at this stage.