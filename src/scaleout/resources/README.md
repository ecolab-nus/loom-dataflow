# Resources
Resources are the lowest-level primitives that constitute the hardware model.
Each resource has a clear creation contract and a consumption/release model.

## Primitive resources

### Memory (On-chip SRAM)
- MemoryCapacity: models on-chip scratchpad/buffer capacity; created with a total byte size; consumed via `consume(size_t)` and freed via `release(size_t)`. Query with `getTotalSize()`, `getAvailableSize()`, `getUsedSize()`, `getUtilizationPercentage()`, and `canConsume(size_t)`.
- MemoryPort: models on-chip SRAM access ports; created with a `PortType` and `port_width` (bits); acquired via `acquire()` and freed via `release()`. Query with `isAvailable()`, `getPortWidth()`, `getPortType()` and `getPortTypeString()`.

### Dataflow
- Ring: on-chip ring/torus interconnect among cores; created without parameters; can only be consumed as a whole via `consume()` and freed via `release()`.
- Chain: on-chip daisy-chain interconnect among cores; created without parameters; can only be consumed as a whole via `consume()` and freed via `release()`.