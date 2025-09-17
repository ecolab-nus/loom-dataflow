# Passes (`lib/passes`)

This directory contains all MLIR-based transformations and helper analyses shipped with TMD. The code is grouped by concern so the command-line tools under `tool/` can link only what they need.

## Structure
- `affine/` – utilities and passes that target affine IR.
- `triton-shared/` – passes specific to Triton-shared lowered kernels.
- `common/` – shared spatial-mapping utilities consumed by both pipelines.
- `input_sharing_analysis.cpp` – standalone affine reuse analysis reused by several tools.

## Affine utilities (`affine/`)
- `affine_tile.{h,cpp}` / `affine_tile_pass.cpp` – tile the outermost `affine.parallel` into perfectly nested loops or turn tiles into standalone passes.
- `affine_parallel_to_for.{h,cpp}` – rewrite an outermost `affine.parallel` into a chain of `affine.for` loops with configurable iterator order.
- Driver mains in `tool/affine/` expose tiling, exploration, and analysis flows.

## Triton-shared passes (`triton-shared/`)
- `triton_shared_affinize.{h,cpp}` – normalize Triton-shared kernels by retyping arguments to `index`, rebuilding affine-friendly arithmetic, replacing eligible `memref` ops with affine variants, and stripping redundant casts.
- `triton_shared_grid_to_parallel.{h,cpp}` – wrap kernels in a 3-D `affine.parallel`, wire grid dimensions into loop bounds, and erase explicit grid index arguments.
- `triton_shared_to_affine.cpp` (in `tool/`) composes these passes with spatial exploration to produce DF-annotated kernels.

## Common utilities (`common/`)
- `spatial_mapping.{h,cpp}` – parse DF modules (`df.spatial_dim`), enumerate spatial mappings for affine loops or Triton kernels, and stitch results back together (including helpers for affine canonicalization).

## Input sharing analysis
`input_sharing_analysis.cpp` implements a textual analysis that reports reuse opportunities for `affine.load`s relative to enclosing loops. The `affine_analyze` tool links it together with tiling utilities and prints annotated IR plus reuse statistics.

See `tool/README.md` for usage examples that combine these components into end-to-end pipelines.
