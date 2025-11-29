# TMD

TMD is an MLIR-backed sandbox for exploring hardware scale-out models, a custom `df` (dataflow) dialect, and compiler passes that bridge Triton-style GPU kernels with affine IR. The repository combines C++ libraries that model hardware resources, MLIR dialect definitions, analysis/transform passes, and command-line tools for experimenting with mappings.

## Highlights
- Custom MLIR dialect `df` to describe spatial dimensions and interconnect topologies.
- C++ runtime for modeling hardware resources (rings, chains, SRAM banks) and higher-level modules (meshes, tori).
- MLIR passes that affinize Triton-shared kernels, tile affine loops, and enumerate spatial mappings.
- Standalone tooling for running the passes and composing dataflow descriptions with transformed kernels.
- Example MLIR programs and exploratory tests under `test/`.

## Repository Map
- `lib/` – C++ libraries and dialect code.
  - `resources/` – primitive hardware resources and manager.
  - `modules/` – compositions such as meshes and tori built from resources.
  - `dataflow-dialect/` – TableGen + C++ for the `df` MLIR dialect.
  - `passes/` – affine and Triton-shared transformations plus shared analyses.
- `tool/` – command-line drivers that wire the passes into runnable pipelines and demos.
- `test/` – GoogleTest unit tests and MLIR inputs covering dialects and passes.
- `build.sh` – release build helper that configures MLIR paths and invokes Ninja.
- `setup_ide.sh` – debug build + `compile_commands.json` generator for IDEs.
- `Testing/` – generated CTest metadata (appears after running CMake).

Detailed documentation for each subsystem lives alongside the code (see the READMEs under `lib/`, `tool/`, and `test/Passes/…`).

## Requirements
- CMake ≥ 3.20, Ninja, a C++17 compiler, and `lld` (or another linker if you override `LLVM_USE_LINKER`).
- An installed LLVM/MLIR build that exports CMake packages. The scripts default to `MLIR_DIR=/opt/llvm-mlir/lib/cmake/mlir`.
- `lit` or `llvm-lit` on `PATH` for CTest-driven MLIR tests (`pipx install lit` is the recommended route).

Quick install commands:

Linux (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install cmake build-essential ninja-build lld
```

Arch Linux
```bash
sudo pacman -S cmake gcc make ninja lld
```

macOS
```bash
brew install cmake ninja
```

Windows
- Visual Studio 2019 or newer with the C++ workload.
- CMake from https://cmake.org/download/.

### Building LLVM/MLIR (quick reference)
```bash
git clone https://github.com/llvm/llvm-project.git $HOME/llvm-project
cd $HOME/llvm-project && mkdir build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
  -DCMAKE_INSTALL_PREFIX=/opt/llvm-mlir \
  -DLLVM_BUILD_UTILS=ON -DLLVM_INSTALL_UTILS=ON
cmake --build . --target check-mlir
ninja install
```

## Build & Configure

### Quick build
```bash
./build.sh
```

Flags such as `--mlir-dir=/path/to/mlir` and `--llvm-lit=/path/to/lit` override defaults. Run `./build.sh --help` for the full list.

### Manual CMake invocation
```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/opt/llvm-mlir/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$(command -v lit || command -v llvm-lit) \
  -DLLVM_USE_LINKER=lld
cmake --build . --config Release
```

### IDE/Debug setup
`./setup_ide.sh` performs a clean Debug build and emits `build/compile_commands.json` for IntelliSense.

## Running Tools & Passes
All binaries live under `build/tool/` after a build. Useful entry points include:
- `triton-shared/single_stage/affinize` – run the Triton-shared affinization pass.
- `triton-shared/single_stage/grid_to_parallel` – replace grid indices with a 3-D `affine.parallel`.
- `triton-shared/single_stage/explore_mapping` – enumerate spatial mappings and merge a DF module.
- `triton-shared/single_stage/annotate_reuse` – attach `tmd.reuse` on `memref.reinterpret_cast`.
- `triton-shared/single_stage/explore_alloc_copy_mapping` – enumerate `memref.alloc`/`memref.copy` mapping choices.
- `ttshared-opt` – end-to-end Triton-shared → Affine/Dataflow pipeline driver.
- `affine_explore`, `affine_tile`, `affine_analyze` – affine-only exploration, tiling, and reuse analysis utilities.

### Triton-shared → Dataflow pipeline (step-by-step)
The repository includes a runnable pipeline that lowers a Triton-shared kernel (already bufferized) into a custom Affine/Dataflow form. The examples under `test/Passes/mm_2Dmesh/` can be reproduced with the following commands executed from the repo root after a build.

#### Option A E2E(Recommended)
```bash
build/tool/ttshared-opt \
  --ttshared test/Triton/mm_fixed_strides/runs/block_64x64x64/ttshared.mlir \
  --df test/Dialect/DataflowDialect/2D_mesh.mlir \
  --dump-dir test/Passes/mm_2Dmesh/ \
  --skip-tile-scf-for-to-l1
```
#### Option B step-by-step
1) Affinize Triton-shared indices
```bash
build/tool/triton-shared/single_stage/affinize \
  --ttshared test/Triton/mm_fixed_strides/runs/block_64x64x64/ttshared.mlir \
  > test/Passes/mm_2Dmesh/01_after_affinization.mlir
```

2) Replace grid indices with a 3-D `affine.parallel`
```bash
build/tool/triton-shared/single_stage/grid_to_parallel \
  --input test/Passes/mm_2Dmesh/01_after_affinization.mlir \
  > test/Passes/mm_2Dmesh/02_after_grid_to_parallel.mlir
```

3) Enumerate spatial mappings and merge DF declarations
```bash
build/tool/triton-shared/single_stage/explore_mapping \
  --input test/Passes/mm_2Dmesh/02_after_grid_to_parallel.mlir \
  --df test/Dialect/DataflowDialect/2D_mesh.mlir \
  > test/Passes/mm_2Dmesh/03_after_exploration.mlir
```

4) Annotate reuse on `memref.reinterpret_cast`
```bash
build/tool/triton-shared/single_stage/annotate_reuse \
  --input test/Passes/mm_2Dmesh/03_after_exploration.mlir \
  > test/Passes/mm_2Dmesh/04_after_reuse_annotation.mlir
```

5) Explore alloc/copy mapping choices
```bash
build/tool/triton-shared/single_stage/explore_alloc_copy_mapping \
  --input test/Passes/mm_2Dmesh/04_after_reuse_annotation.mlir \
  > test/Passes/mm_2Dmesh/05_after_memref_mapping.mlir
```

6) Bufferize tensors to memrefs
```bash
mlir-opt \
  --one-shot-bufferize="allow-unknown-ops allow-return-allocs-from-loops" \
  test/Passes/mm_2Dmesh/05_after_memref_mapping.mlir \
  > test/Passes/mm_2Dmesh/06_after_bufferization.mlir
```

7) Tile scf.for loops to fit L1 (optional)
```bash
build/tool/triton-shared/single_stage/tile_scf_for_to_l1 \
  test/Passes/mm_2Dmesh/06_after_bufferization.mlir \
  > test/Passes/mm_2Dmesh/07_after_for_tiling.mlir
``` 

Notes:
- The end-to-end driver accepts `--map-analysis-only` to attach `tmd.copy.candidates` without cloning functions.
- The single-stage alloc/copy explorer accepts `--analysis-only` with the same effect.

### Pass reference (purpose, limitations, implementation)
- Affinize Triton-shared indices (`tmd-triton-shared-affinize`)
  - Purpose: Rewrite arithmetic index expressions into `affine.apply`, convert eligible loads/stores to affine form, and express `memref.reinterpret_cast` offsets via affine maps. Treats trailing grid/thread arguments as dims/symbols to expose GPU-style indexing to affine.
  - Limitations: Conservative—only provably affine expressions are converted. Assumes the last 6 function arguments encode grid sizes/indices; nonconforming kernels are left unchanged. Some `memref` ops remain non-affine if indices are not proven affine.
  - Implementation: See `lib/passes/triton-shared/triton_shared_affinize.{h,cpp}`; pass argument is `tmd-triton-shared-affinize`.

- Grid-to-parallel (`tmd-triton-shared-grid-to-parallel`)
  - Purpose: Replace the last three grid index arguments with a 3-D `affine.parallel` with dynamic uppers `(sizeX,sizeY,sizeZ)`; erase index args from the signature and replace their uses by the parallel IVs.
  - Limitations: Requires ≥ 6 function args following `(sizeX,sizeY,sizeZ, idxX,idxY,idxZ)`; otherwise no-op. Expects sizes to be of `index` (affinization establishes this in typical flows).
  - Implementation: See `lib/passes/triton-shared/triton_shared_grid_to_parallel.{h,cpp}`.

- Spatial mapping exploration
  - Purpose: Enumerate mappings from hardware `df.spatial_dim` declarations to the outermost `affine.parallel` iterators; clone per mapping, annotate inner loops with `tmd.mapped_to`, and insert outer `affine.for` “waves” when the mesh cannot cover the grid in one shot.
  - Limitations: Combinatorial growth in clones due to partitioning/permutation of dims and outer-for orderings. Exploration is structural (not resource-capacity aware) in this prototype.
  - Implementation: `lib/passes/triton-shared/spatial_mapping.{h,cpp}` (`EnumerateSpatialMappings`). CLI: `build/tool/triton-shared/single_stage/explore_mapping`.

- Reuse annotation on reinterpret-cast (`tmd-annotate-reinterpret-cast-reuse`)
  - Purpose: Attach a `tmd.reuse` dictionary to each `memref.reinterpret_cast` describing how its offset varies with surrounding iterators, grouped by `spatial` (`affine.parallel`), `temporal` (`affine.for`), and `sequential` (`scf.for`). Each entry records `iterator` (SSA name), `depth`, `reuse_type` (`no_reuse`/`total_reuse`), `volume` (bytes; 0, full block, or -1 unknown), and `mapped_to` for spatial entries.
  - Limitations: Binary reuse classification only (no partial reuse yet). Volumes require known block sizes.
  - Implementation: `lib/passes/triton-shared/reinterpret_cast_reuse.{h,cpp}`. CLI: `build/tool/triton-shared/single_stage/annotate_reuse`.

- Alloc/Copy mapping exploration (`tmd-explore-alloc-copy-mapping`)
  - Purpose: Annotate `memref.alloc` with `{tmd.alloc={local=true, memory_name=…}}` and enumerate per-`memref.copy` mapping choices: local memory copies and broadcasts along dimensions with spatial total-reuse. Merge the DF module to discover one `df.memory` and classify `df.interconnects` as x/y based on affine maps.
  - Limitations: Assumes a single `df.memory`; interconnect classification is heuristic (e.g., `(d0+1,d1)`→x, `(d0,d1+1)`→y). Enumerating the cross-product of candidates can explode; use analysis-only when needed.
  - Implementation: `lib/passes/triton-shared/explore_alloc_copy_mapping.{h,cpp}` and notes in `lib/passes/triton-shared/README.alloc_copy_mapping.md`. CLI: `build/tool/triton-shared/single_stage/explore_alloc_copy_mapping` (or end-to-end via `build/tool/ttshared-opt`).

## Tests & Examples
```bash
cd build
ctest --output-on-failure
```

The `test/` folder contains MLIR snippets that showcase dialect usage (e.g., `test/Dialect/DataflowDialect/2D_mesh.mlir`) and Triton MM variations. `test/Passes/explore_over_triton_shared` documents how to reproduce the exploration pipeline end-to-end.

## Troubleshooting
- `lit` not found: install via `pipx install lit` (preferred) or provide `--llvm-lit=/path/to/lit`.
- `MLIRConfig.cmake` missing: export `MLIR_DIR` to point at your LLVM/MLIR installation.
- IntelliSense gaps: rerun `./setup_ide.sh` so that `compile_commands.json` stays in sync with headers generated by TableGen.

## License
See `LICENSE`.
