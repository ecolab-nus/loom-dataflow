# Alloc/Copy Mapping Exploration (Prototype)

## Problem
After the spatial mapping and reuse analysis, kernels still contain generic `memref.alloc` and `memref.copy`. We need to indicate how these map to the hardware dataflow fabric:
- Where allocations live (local memory tiles)
- How copies are realized: local memory loads vs. broadcasts over interconnects

## Assumptions
- Single `df.memory` handle (local memory pool)
- Optional `df.interconnects` that connect memory tiles horizontally/vertically
- `tmd.reuse` is already attached to relevant `memref.reinterpret_cast` ops

## Attributes
- On `memref.alloc`:
  - `tmd.alloc = { local = true, memory_name = "<df.memory name>" }`

- On `memref.copy`:
  - Analysis-only (no clones):
    - `tmd.copy.candidates = [ {kind=mem|broadcast, ...}, ... ]`
  - Enumerated clones (per variant):
    - `tmd.copy.choice = {kind=mem|broadcast, ... }` (exactly one)

Broadcast candidates require spatial total-reuse along the matching dimension (`tmd.reuse.spatial[*].reuse_type == "total_reuse"` and `mapped_to` = dimension name).

## Pass
CLI: `tmd-explore-alloc-copy-mapping`

Modes:
- Analysis-only: attach `tmd.copy.candidates` in-place
- Enumeration: clone per cross-product of candidates, attach `tmd.copy.choice`

Function names are suffixed to reflect choices, e.g. `__c0mem_c1bx_c2by`.


