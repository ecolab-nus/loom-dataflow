# Passes (`lib/passes`)

This directory contains all MLIR-based transformations and helper analyses shipped with TMD. The code is grouped by concern so the command-line tools under `tool/` can link only what they need.

## Structure
- `affine/` – utilities and passes that target affine IR.
- `triton-shared/` – passes specific to Triton-shared lowered kernels.
- `common/` – shared spatial-mapping utilities and analyses (including `input_sharing_analysis.cpp`) consumed by both pipelines.

## Affine utilities (`affine/`)
- `affine_tile.{h,cpp}` / `affine_tile_pass.cpp` – tile the outermost `affine.parallel` into perfectly nested loops or turn tiles into standalone passes. Used both directly and as a building block by the spatial mapping routines.
- `affine_parallel_to_for.{h,cpp}` – rewrite an outermost `affine.parallel` into a chain of `affine.for` loops with configurable iterator order, so spatial exploration can enumerate wave permutations.
- Driver mains in `tool/affine/` expose tiling (`affine_tile`), exploration (`affine_explore`), and reuse analysis (`affine_analyze`) flows that stitch these utilities together.

## Triton-shared passes (`triton-shared/`)
#### Input 
Triton emits `tt.shared` kernels that expect to run on a GPU grid; the last six function arguments encode the launch grid extents and the current program IDs (`program_id.{x,y,z}`). Example of such input can be found in `test/Dialect/Triton/mm_fixed_strides/ttshared.mlir`.
#### Passes 
the pass pipeline `build/tool/ttshared-opt` convert this `ttshared.mlir` through 5 stages: 
- **affinization**: try to convert the arith operations in the ttshared into affine formulas
- **grid_to_parallel**: convert the grid representation in the orginal ttshared into `afffine.parallel` representations, where the used grid dimensions become the parallel for loop
- **explore_mapping**: tile and reorder the `affine.parallel` loops
- **annotate_reuse**: calculate the data reuse among spatial cores and annotate the reuse volume
    

## Common utilities (`common/`)
- `spatial_mapping.{h,cpp}` – parse DF modules (`df.spatial_dim`), enumerate spatial mappings for affine loops or Triton kernels, and stitch results back together (including helpers for affine canonicalization). Cloned functions are suffixed with tokens `d<dimIndex>i<iterIndex>` (mapping dimension `dimIndex` to iterator `iterIndex`) and, when iterator orders are permuted, additional `_f<pos>` tokens to record the chosen order. This mirrors the notation visible in examples such as `@matmul_kernel__d0i0_d1i0_f0_f1`.
- `reinterpret_cast_reuse.{h,cpp}` – annotate `memref.reinterpret_cast` ops with a `tmd.reuse` attribute that captures whether the slice offset varies with each surrounding spatial (`affine.parallel` ↦ dataflow-parallel cores), temporal (`affine.for` ↦ wave sequencing across the fabric), or sequential (`scf.for` ↦ per-core tile loop) iterator. Each entry records the iterator SSA name, nesting depth, a `reuse_type` (`no_reuse` or `total_reuse`, with partial reuse reserved for future work), and the amount of data reused (`volume`, currently 0 or the entire block size, or -1 when unknown). A `mapped_to` field is kept for spatial iterators so the hardware dimension is explicit.
- `input_sharing_analysis.cpp` – textual analysis that reports reuse opportunities for `affine.load`s relative to enclosing loops. The `affine_analyze` tool links it together with tiling utilities and prints annotated IR plus reuse statistics.

See `tool/README.md` for usage examples that combine these components into end-to-end pipelines.
