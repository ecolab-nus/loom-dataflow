# Passes (`lib/passes`)

This directory contains all MLIR-based transformations and helper analyses shipped with LOOM. The code is grouped by concern so the command-line tools under `tool/` can link only what they need.

## Structure
- `affine/` – utilities and passes that target affine IR.
- `loom-opt/` – passes specific to Triton-shared lowered kernels.
- `common/` – shared driver helpers, IR utilities, and analyses consumed by both pipelines.

## Affine utilities (`affine/`)
- `affine_tile.{h,cpp}` / `affine_tile_pass.cpp` – tile the outermost `affine.parallel` into perfectly nested loops or turn tiles into standalone passes.
- `affine_parallel_to_for.{h,cpp}` – rewrite an outermost `affine.parallel` into a chain of `affine.for` loops with configurable iterator order.
- Driver mains in `tool/affine/` expose tiling (`affine_tile`), exploration (`affine_explore`), and reuse analysis (`affine_analyze`) flows that stitch these utilities together.

## Triton-shared passes (`loom-opt/`)
#### Input 
Triton emits `tt.shared` kernels that expect to run on a GPU grid; the last six function arguments encode the launch grid extents and the current program IDs (`program_id.{x,y,z}`). Example of such input can be found in `test/Dialect/Triton/mm_fixed_strides/ttshared.mlir`.
#### Passes 
the pass pipeline `build/tool/ttshared-opt` (deprecated) convert this `ttshared.mlir` through 5 stages: 
- **affinization** (deprecated): try to convert the arith operations in the ttshared into affine formulas
- **grid_to_parallel** (deprecated): convert the grid representation in the orginal ttshared into `afffine.parallel` representations, where the used grid dimensions become the parallel for loop
- **spatial_mapping**: copy ADL hardware declarations into the pipeline IR
- **analyze_reuse**: analyze the data reuse among spatial cores and annotate the reuse volume
    

## Common utilities (`common/`)
- `driver_utils.{h,cpp}` – shared command-line helpers, including ADL dialect registration, MLIR parsing/printing, and copying `@arch_system` declarations into merged pipeline modules.
- `analyze_reuse.{h,cpp}` – analyze and annotate `memref.reinterpret_cast` ops with a `loom.reuse` attribute that captures whether the slice offset varies with each surrounding spatial (`affine.parallel` ↦ dataflow-parallel cores), temporal (`affine.for` ↦ wave sequencing across the fabric), or sequential (`scf.for` ↦ per-core tile loop) iterator. Each entry records the iterator SSA name, nesting depth, a `reuse_type` (`no_reuse` or `total_reuse`, with partial reuse reserved for future work), and the amount of data reused (`volume`, currently 0 or the entire block size, or -1 when unknown). A `mapped_to` field is kept for spatial iterators so the hardware dimension is explicit.
- `src/deprecated/input_sharing_analysis.cpp` (deprecated) – textual analysis that reports reuse opportunities for `affine.load`s relative to enclosing loops.

See `tool/README.md` for usage examples that combine these components into end-to-end pipelines.
