# Tools (tmd/tool)

Command-line drivers that exercise passes and demos. All executables are built under `build/tool/`.

## Triton-shared pipeline and tools

- `triton_shared_to_affine`
  - Pipeline runner for Triton-shared kernels.
  - Runs: Triton-shared affinization → grid-to-parallel → spatial mapping enumeration (and outer-for exploration) and merges DF declarations.
  - Usage:
    ```bash
    build/tool/triton_shared_to_affine \
      --ttshared path/to/ttshared.mlir \
      --df path/to/df.mlir > merged.mlir
    ```

- `triton_shared_affinize`
  - Runs the Triton-shared affinization pass alone on a single MLIR module from stdin or file.
  - Usage:
    ```bash
    build/tool/triton_shared_affinize input.mlir > output.mlir
    ```

- `triton_shared_grid_to_parallel`
  - Wraps functions with a 3-D `affine.parallel` using the trailing size args and removes the trailing index args.
  - Usage:
    ```bash
    build/tool/triton_shared_grid_to_parallel input.mlir > output.mlir
    ```

- `triton_shared_explore`
  - Enumerates spatial mappings for a Triton-shared-after-grid-to-parallel module and merges DF plus generated clones.
  - Usage:
    ```bash
    build/tool/triton_shared_explore \
      --ttshared path/to/ttshared.mlir \
      --df path/to/df.mlir > merged.mlir
    ```

## Affine tools

- `affine_explore`
  - Enumerates mappings of DF spatial dimensions to the iterators of the first outermost `affine.parallel` in each function of an Affine module, producing function clones per mapping and merging with DF declarations.
  - Usage:
    ```bash
    build/tool/affine_explore \
      --affine path/to/affine.mlir \
      --df path/to/df.mlir > merged.mlir
    ```

- `affine_tile`
  - Tiles the first outermost `affine.parallel` by a given factor along a given iterator.
  - Usage:
    ```bash
    build/tool/affine_tile input.mlir [tiling_factor] [tile_dim_index] > output.mlir
    ```

- `affine_analyze`
  - Runs input sharing and reuse analyses over Affine IR; prints annotated IR and a text report to stdout.
  - Usage:
    ```bash
    build/tool/affine_analyze input.mlir > annotated.mlir
    ```

## Dataflow-dialect utility

- `df_parse_print`
  - Parses DF IR fragments and prints them back for quick verification.
  - Usage:
    ```bash
    build/tool/df_parse_print input.mlir
    ```

## Resource-system demos

- `resource_demo`
  - Demonstrates resource types (`MemoryPort`, `MemoryCapacity`, `ResourceManager`) and simple acquisition/consumption flows.
  - Usage:
    ```bash
    build/tool/resource_demo
    ```

- `resource_module_demo`
  - Demonstrates module-level resources and acquisition over `Torus`, `Mesh2D`, and `Chain` with simple topologies.
  - Usage:
    ```bash
    build/tool/resource_module_demo
    ```
