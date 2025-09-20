# Passes (`lib/passes`)

This directory contains all MLIR-based transformations and helper analyses shipped with TMD. The code is grouped by concern so the command-line tools under `tool/` can link only what they need.

## Structure
- `affine/` – utilities and passes that target affine IR.
- `triton-shared/` – passes specific to Triton-shared lowered kernels.
- `common/` – shared spatial-mapping utilities and analyses (including `input_sharing_analysis.cpp`) consumed by both pipelines.

## Affine utilities (`affine/`)
- `affine_tile.{h,cpp}` / `affine_tile_pass.cpp` – tile the outermost `affine.parallel` into perfectly nested loops or turn tiles into standalone passes.
- `affine_parallel_to_for.{h,cpp}` – rewrite an outermost `affine.parallel` into a chain of `affine.for` loops with configurable iterator order.
- Driver mains in `tool/affine/` expose tiling, exploration, and analysis flows.

## Triton-shared passes (`triton-shared/`)
- **Input model.** Triton emits `tt.shared` kernels that expect to run on a GPU grid; the last six function arguments encode the launch grid extents and the current program IDs (`program_id.{x,y,z}`). The files under `test/Dialect/Triton/mm_fixed_strides/` (notably `ttshared.mlir`) capture the IR directly from Triton.
- `triton_shared_affinize.{h,cpp}` – normalize Triton-shared kernels by retyping arguments to `index`, rebuilding affine-friendly arithmetic, replacing eligible `memref` ops with affine variants, and stripping redundant casts. The goal is to expose the same indexing logic as affine expressions so later passes can reason about iteration spaces.
- `triton_shared_grid_to_parallel.{h,cpp}` – wrap kernels in a 3-D `affine.parallel`, wire grid dimensions into loop bounds, and erase explicit grid index arguments. After this stage, the outermost `affine.parallel` enumerates all GPU grid coordinates.
- `triton_shared_to_affine.cpp` (in `tool/`) composes these passes with spatial exploration to produce DF-annotated kernels. A hardware description written in the evolving `df` dialect (example: `test/Dialect/DataflowDialect/2D_mesh.mlir`) declares named spatial dimensions. The exploration clones each kernel for every viable mapping between hardware dimensions and the grid loops, annotating mapped `affine.parallel` iterators with `tmd.mapped_to` and emitting additional `affine.for` loops when execution must be time-multiplexed (“waves”) across the mesh. Original `scf.for` loops remain to represent per-core sequencing of tiles.

## Common utilities (`common/`)
- `spatial_mapping.{h,cpp}` – parse DF modules (`df.spatial_dim`), enumerate spatial mappings for affine loops or Triton kernels, and stitch results back together (including helpers for affine canonicalization).
- `reinterpret_cast_reuse.{h,cpp}` – annotate `memref.reinterpret_cast` ops with a `tmd.reuse` attribute that captures whether the slice offset varies with each surrounding spatial (`affine.parallel` ↦ dataflow-parallel cores), temporal (`affine.for` ↦ wave sequencing across the fabric), or sequential (`scf.for` ↦ per-core tile loop) iterator. Each entry records the iterator SSA name, nesting depth, a `reuse_type` (`no_reuse` or `total_reuse`, with partial reuse reserved for future work), and the amount of data reused (`volume`, currently 0 or the entire block size, or -1 when unknown). A `mapped_to` field is kept for spatial iterators so the hardware dimension is explicit.
- `input_sharing_analysis.cpp` – textual analysis that reports reuse opportunities for `affine.load`s relative to enclosing loops. The `affine_analyze` tool links it together with tiling utilities and prints annotated IR plus reuse statistics.

See `tool/README.md` for usage examples that combine these components into end-to-end pipelines.
