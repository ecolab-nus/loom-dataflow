# Modules

Modules describe reusable hardware topologies that are assembled from primitive resources. They do not create their own resources; instead they reference instances from `lib/resources/` and keep track of which cores are linked via `mlir::AffineMap` placement metadata.

## Core abstractions
- `Module` – lightweight base class that stores a name and exposes `getTypeName()`.
- `NetworkModule` – extends `Module` with a list of participating core ids and a mandatory affine placement map. All topology-aware modules inherit from it.

## Provided modules
- `Chain` – wraps a `resources::Chain` interconnect to form a 1-D topology. Offers `isAvailable()`, `acquire()`, and `release()` helpers.
- `Mesh2D` – composes horizontal and vertical `Chain` instances to model a rectangular mesh. Acquisition ensures every underlying chain is reserved atomically.
- `Torus` – wraps a `resources::Ring` and optional per-core memory capacities or memory banks. Provides broadcast/reduction primitives (`canBroadcast`, `acquireForBroadcast`, `releaseBroadcast`, `acquireForReduction`, `releaseReduction`).

All modules are constructed with:
1. A name for logging/debugging.
2. The ordered list of core ids participating in the topology.
3. An affine map that translates logical coordinates to physical cores.
4. References to the resources they consume.

## Integration notes
- Placement maps (`mlir::AffineMap`) make it possible to connect module descriptions to MLIR analyses, enabling the passes under `lib/passes/` to align DF descriptions with affine loops.
- Memory-aware modules (currently `Torus`) support either capacity-only tracking or grouped port+capacity accounting through `resources::MemoryBank`.
- Future modules can derive from `NetworkModule` to reuse placement bookkeeping and interoperate with existing tooling.
