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
- `spatial_mapping` – merge ADL declarations from a hardware spec into a Triton-shared module.
  ```bash
  build/tool/loom-opt/single_stage/spatial_mapping \
    --input path/to/input.mlir \
    --hw_spec path/to/adl.mlir > merged.mlir
  ```
- `ttshared-opt` (deprecated) – end-to-end pipeline: affinize → grid-to-parallel → spatial exploration. 
  ```bash
  # Note: This tool is no longer built by default. Implementation preserved in tool/loom-opt/single_stage/ deprecated/
  ```

The pipeline expects a Triton `tt.shared` kernel (see `test/Dialect/Triton/mm_fixed_strides/ttshared.mlir`) together with an ADL hardware description. Stage 3 currently prepends declarations from the spec's `@arch_system` module to the transformed kernel IR; hardware mapping enumeration is intentionally left for a future rewrite.

The reuse annotator attaches a `loom.reuse` dictionary to each `memref.reinterpret_cast`. Entries are grouped by iterator kind: `spatial` for hardware-mapped `affine.parallel` loops when present, `temporal` for `affine.for` loops that schedule successive waves across the fabric, and `sequential` for per-core `scf.for` loops that step through tiles locally.

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
