# Tools (`tool/`)

Executables built from this directory wrap the passes in `lib/passes/` and expose them as command-line utilities. After building the project, binaries appear under `build/tool/` with the same names described below.

## Triton-shared pipeline
- `triton_shared_affinize` – run the Triton-shared affinization pass only.
  ```bash
  build/tool/triton_shared_affinize input.mlir > output.mlir
  ```
- `triton_shared_grid_to_parallel` – wrap kernels in a 3-D `affine.parallel` and drop explicit grid indices.
  ```bash
  build/tool/triton_shared_grid_to_parallel input.mlir > output.mlir
  ```
- `triton_shared_explore` – enumerate spatial mappings for a Triton-shared kernel after grid-to-parallel and merge DF declarations provided via `--df`.
  ```bash
  build/tool/triton_shared_explore \
    --ttshared path/to/ttshared.mlir \
    --df path/to/df.mlir > merged.mlir
  ```
- `triton_shared_to_affine` – end-to-end pipeline: affinize → grid-to-parallel → spatial exploration. Produces a merged module that contains the DF module and generated clones.
  ```bash
  build/tool/triton_shared_to_affine \
    --ttshared path/to/ttshared.mlir \
    --df path/to/df.mlir > merged.mlir
  ```

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
