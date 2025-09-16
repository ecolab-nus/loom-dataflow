# Passes (tmd/lib/passes)

This directory contains all MLIR passes and helper algorithms used by the project. It is organized into three subdirectories by concern, plus this README:

- `affine/`: Utilities and helpers for working with Affine loops and maps.
  - `affine_tile.h/.cpp`: Implements tiling of `affine.parallel` into perfectly nested outer/inner loops.
  - `affine_parallel_to_for.h/.cpp`: Converts an outermost `affine.parallel` into a nested chain of `affine.for` loops according to a chosen iterator order.
- `triton-shared/`: Pass implementations targeting Triton-shared lowered kernels.
  - `triton_shared_affinize.h/.cpp`: Affinizes index math and converts eligible memory ops to affine forms.
  - `triton_shared_grid_to_parallel.h/.cpp`: Wraps function bodies with 3-D `affine.parallel` using the trailing size arguments and removes the trailing index arguments.
- `common/`: Shared analysis/enumeration utilities used by both Affine and Triton-shared flows.
  - `spatial_mapping.h/.cpp`: Spatial-dimension collection from the Dataflow (DF) module and enumeration of spatial mappings; also includes helpers to compose/canonicalize affine.apply.

The actual pass drivers (standalone command-line tools) live under `tmd/tool/` and link these sources directly. See `tmd/tool/README.md` for usage.

## Triton Shared Affinize Pass

This pass converts arithmetic index expressions produced by Triton-shared lowered kernels into affine form and replaces eligible memory and loop constructs with their affine counterparts. The goal is to maximize the use of `affine.apply`, `affine.min/max`, `affine.load/store`, and index-typed IR, while eliminating redundant casts and dead arithmetic.

### Summary of transformations

- **Function argument retyping**
  - Convert all `i32` function arguments to `index` to eliminate pervasive `arith.index_cast` in the body and enable direct use in affine expressions.

- **Integer arithmetic harmonization**
  - Rebuild `arith.addi/subi/muli/divsi/minsi/maxsi` where any operand/result is `index` or mixed-typed so that operands/results are of type `index`.
  - Iterate to a fixed point to remove all mixed-type arithmetic introduced by retyping.

- **Index → `affine.apply` conversion**
  - For index-typed SSA values representing affine expressions, inject `affine.apply` and feed users with its result.
  - Partial conversions for `addi/subi` when one side is affine: the affine side is folded into `affine.apply`, the other side remains as-is.

- **`memref.reinterpret_cast` offsets affinization**
  - Rebuild the op with offsets expressed via `affine.apply` when the offset expressions are affine in terms of function args and surrounding loop IVs.
  - Sizes/strides are kept unchanged (focus is offsets) to avoid dominance and aliasing issues.

- **`tensor.extract_slice` / `memref.subview` sizes affinization**
  - Convert dynamic sizes to `affine.apply` where possible.
  - Recognize and replace common clamp patterns with a single `affine.min` over a multi-result affine map:
    - Pattern: `min( max(min(base + T, dim), base) - base, T )` → `affine.min( dim - base, T )`.
  - For `min/max` of affine expressions, build a multi-result affine map and emit `affine.min`/`affine.max` directly.

- **`memref.load/store` → `affine.load/store`**
  - Replace when all indices are either already trivially affine (loop IVs / block args / constant indices) or can be converted to `affine.apply` via the index → affine conversion.

- **`scf.for` normalization (index IV and affine bounds)**
  - Rebuild `scf.for` with index-typed lower bound, upper bound, and step. Clone the loop body so the induction variable is natively type `index` without extra casts.
  - Affinize loop bounds and step with `affine.apply`/`affine.min/max` where possible, so upper bounds can be expressed as affine maps.

- **Redundant cast cleanup and DCE**
  - Remove redundant `arith.index_cast` where source and destination types are the same.
  - General dead code elimination: iteratively erase any trivially-dead operation (using MLIR’s `isOpTriviallyDead`).
  - Iteration guards ensure termination even if follow-on canonicalizations unlock additional cleanup.

### Affine-eligibility and expression model

Dims: function block arguments and surrounding `scf.for` IVs. Symbols: other index values that cannot be proven as dims. The pass permits `addi/subi`, mul by constant, div by non-zero constant (`floorDiv`), constants, and `arith.index_cast` passthrough. Non-affine pieces are promoted to symbols.

### Dialects and dependencies

Requires: `affine`, `arith`, `memref`, `func`, `scf`, `linalg`, `tensor`, `bufferization`.

### Running the pass

- Pass name: `tmd-triton-shared-affinize`
- As a standalone tool in this repo (built under `build/tool/`):

```bash
build/tool/triton_shared_affinize input.mlir > output.mlir
```

## Triton Shared Grid→Parallel Pass

Exposes the implicit grid parallelism as a 3-D `affine.parallel` and simplifies the kernel signature by removing the three grid index arguments.

### Transformation

- Input convention (last 6 args): `(sizeX, sizeY, sizeZ, idxX, idxY, idxZ)`.
- Inserts a single 3-D `affine.parallel` with lower bounds `(0,0,0)`, upper bounds `(sizeX,sizeY,sizeZ)`, and steps `(1,1,1)`.
- Replaces uses of `(idxX, idxY, idxZ)` with the parallel IVs and erases the trailing 3 index arguments.

### Running the pass tool

```bash
build/tool/triton_shared_grid_to_parallel input.mlir > output.mlir
```

## Common: Spatial Mapping Utilities

- Discover spatial dimensions from a DF module (`df.spatial_dim` ops).
- Enumerate mappings of spatial dims to Affine programs or mappings from Triton-shared grid dims to hardware spatial dims.
- Provide helpers to compose and canonicalize `affine.apply` and convert between `affine.parallel` and nested `affine.for`.

## Related Tools (under tmd/tool)

- `triton_shared_to_affine`: Runs the end-to-end pipeline: affinize → grid-to-parallel → mapping enumeration (and optional outer-for exploration). Output is a merged module with DF declarations.
- `triton_shared_explore`: Enumerates spatial mappings over a Triton-shared-after-grid-to-parallel program and merges DF + generated clones.
- `affine_explore`: Enumerates spatial mappings over an Affine program and merges DF + generated clones.

Refer to `tmd/tool/README.md` for complete tool descriptions and CLI flags.

