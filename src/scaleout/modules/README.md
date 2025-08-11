## Modules
Resources are the primitive building blocks (SRAM capacity/ports, rings, chains). **Modules** are higher-level compositions of these resources that represent concrete hardware topologies and behaviors. A module encapsulates:

- what cores it spans
- what resources it requires
- how it consumes/releases those resources when an operation is mapped

Compiler integration (declaring which IR ops can map to a module and with which constraints) is planned via PDLL generation, but is not implemented yet.

## Provided primitive modules

### Torus (ring-backed)
Links a set of cores via a `Ring` interconnect. Core indices should follow an affine function compatible with the ring placement.

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
A 2D mesh of size (N, M) is modeled as N horizontal `Chain`s (left–right) and M vertical `Chain`s (top–down) that connect all cores in a grid.

- Each row/column behaves like a chain (no reduction support).
- Use Torus for reduction-optimized collectives.

## C++ API (initial)
This directory defines minimal C++ classes to model modules:

- `Module`: base class holding a name and the list of core ids.
- `Torus`: wraps a `Ring` and optional per-core `MemoryCapacity` for broadcast accounting. Methods:
  - `canBroadcast(element_bytes)`
  - `acquireForBroadcast(element_bytes)` / `releaseBroadcast(element_bytes)`
  - `acquireForReduction()` / `releaseReduction()`
- `Mesh2D`: wraps horizontal and vertical `Chain`s. Methods:
  - `isAvailable()`
  - `acquire()` / `release()`

Compiler/IR matching hooks are intentionally omitted at this stage.