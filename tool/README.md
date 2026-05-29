# Tools (`tool/`)

Executables built from this directory wrap the passes in `lib/passes/` and expose them as command-line utilities. After building the project, binaries appear under `build/tool/` with the same names described below.

## Loom Opt pipeline
- `affinize` (deprecated) – run the Triton-shared affinization pass only.
  ```bash
  # Note: This tool is no longer built by default. Implementation preserved in tool/loom-opt/single_stage/ deprecated/
  ```
- `grid_to_parallel` (deprecated) – wrap kernels in a 3-D `affine.parallel` and drop explicit grid indices.
  ```bash
  # Note: This tool is no longer built by default. Implementation preserved in tool/loom-opt/single_stage/ deprecated/
  ```
- `enumerate_hw_mapping` – enumerate spatial mappings for a Triton-shared kernel after grid-to-parallel and merge DF declarations provided via `--df`.
  ```bash
  build/tool/loom-opt/single_stage/enumerate_hw_mapping \
    --input path/to/input.mlir \
    --df path/to/df.mlir > merged.mlir
  ```
- `ttshared-opt` (deprecated) – end-to-end pipeline: affinize → grid-to-parallel → spatial exploration. 
  ```bash
  # Note: This tool is no longer built by default. Implementation preserved in tool/loom-opt/single_stage/ deprecated/
  ```

The pipeline expects a Triton `tt.shared` kernel (see `test/Dialect/Triton/mm_fixed_strides/ttshared.mlir`) together with a hardware description written in the nascent `df` dialect (e.g. `test/Dialect/DataflowDialect/2D_mesh.mlir`). After affinization, the GPU launch grid becomes a single 3-D `affine.parallel`. Exploration then pairs those iterators with the declared hardware dimensions, cloning the kernel per mapping, tagging parallel loops with `loom.mapped_to`, and inserting outer `affine.for` loops whenever additional “waves” are required to cover the full grid. Nested `scf.for` loops stay within a core to model sequential tile processing.

Both `enumerate_hw_mapping` and `ttshared-opt` also run the reuse annotator. Look for a `loom.reuse` dictionary attached to each `memref.reinterpret_cast`. Entries are grouped by iterator kind: `spatial` for `affine.parallel` loops that map work across hardware cores, `temporal` for `affine.for` loops that schedule successive waves across the fabric, and `sequential` for per-core `scf.for` loops that step through tiles locally. Each entry lists the induction-variable SSA name (e.g. `%arg13`), the nesting `depth`, a `reuse_type` (`no_reuse` or `total_reuse` for now), and the reused `volume` in bytes (0 for `no_reuse`, the full block size for `total_reuse`, or -1 when the size is not statically known). Partial reuse classification will arrive in a later iteration; spatial entries also keep `mapped_to` to surface the hardware dimension.

## Affine utilities
- `affine_explore` – enumerate mappings between DF spatial dims and the outermost `affine.parallel` loops in an affine module.
  ```bash
  build/tool/affine_explore \
    --affine path/to/affine.mlir \
    --df path/to/df.mlir > merged.mlir
  ```
- `affine_tile` – tile the first outermost `affine.parallel` by a user-specified factor and iterator index.
  ```bash
  build/tool/affine_tile input.mlir 8 0 > tiled.mlir
  ```
- `affine_analyze` – run input-sharing/reuse analysis and print annotated IR alongside a text report.
  ```bash
  build/tool/affine_analyze input.mlir > annotated.mlir
  ```

## Resource-system demos
- `resource_demo` – exercise primitive resources (`MemoryCapacity`, `MemoryPort`, `ResourceManager`).
- `resource_module_demo` – showcase module-level resource acquisition flows (chains, mesh, torus).

All tools require the same MLIR build used by the libraries; the CMake targets automatically link the necessary dialects and support libraries.
