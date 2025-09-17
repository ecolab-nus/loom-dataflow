# Resources

Resources are the primitive building blocks for the scale-out model. Each resource tracks its own availability and implements a clear acquire/release contract through the CRTP `Resource<T>` base class found in `resource_base.h`.

## Memory-focused resources
- `MemoryCapacity` – models on-chip SRAM capacity with byte-accurate accounting. Offers queries (`getTotalSize`, `getAvailableSize`, `getUtilizationPercentage`), guards via `canConsume`, and stateful `consume`/`release` helpers.
- `MemoryPort` – represents a single read, write, or read-write port. Acquisition is exclusive and validated against the supported operation type.
- `MemoryBank` – convenience wrapper that groups one capacity object with optional read/write ports, providing atomic `acquireForTransfer`/`releaseTransfer` semantics.

## Interconnect resources
- `Ring` – models a ring/torus network. Consumption is all-or-nothing and mutually exclusive.
- `Chain` – represents a daisy-chain interconnect with the same availability contract as `Ring`.

## Resource management
`ResourceManager` is a singleton that tracks all instantiated resources. It supports registration, lookup by id, bulk inspection (`getResourceStatistics`), and cleanup—useful for demos and tooling that need visibility into the runtime model.

Each module in `lib/modules/` consumes or checks these resources to mirror how hardware would accept work.
