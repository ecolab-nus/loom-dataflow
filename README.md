<img src="assets/loom-logo.svg" alt="Loom Logi" width="250">


LOOM is an MLIR-backed sandbox for exploring hardware scale-out models, a custom `df` (dataflow) dialect, and compiler passes that bridge Triton-style GPU kernels with affine IR. The repository combines C++ libraries that model hardware resources, MLIR dialect definitions, analysis/transform passes, and command-line tools for experimenting with mappings.

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
- `benchmark.sh` – performance benchmarking script that measures command execution time with statistical analysis.
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
- `triton-shared/single_stage/enumerate_hw_mapping` – enumerate spatial mappings and merge a DF module.
- `triton-shared/single_stage/hoist_block_loading` – hoist block loading operations from innermost loops.
- `triton-shared/single_stage/annotate_reuse` – attach `loom.reuse` on `memref.reinterpret_cast`.
- `triton-shared/single_stage/explore_alloc_copy_mapping` – enumerate `memref.alloc`/`memref.copy` mapping choices.
- `triton-shared/single_stage/tile_scf_for_to_l1` – tile `scf.for` loops to fit L1 memory capacity.
- `ttshared-opt` – end-to-end Triton-shared → Affine/Dataflow pipeline driver.
- `affine_explore`, `affine_tile`, `affine_analyze` – affine-only exploration, tiling, and reuse analysis utilities.


### Triton-shared → Dataflow pipeline (step-by-step)
The repository includes a runnable pipeline that lowers a Triton-shared kernel (already bufferized) into a custom Affine/Dataflow form. The examples under `test/Passes/mm_2Dmesh/` can be reproduced with the following commands executed from the repo root after a build.

<!-- #### Option A E2E(Recommended)
```bash
build/tool/ttshared-opt \
  --ttshared test/Triton/mm_fixed_strides/runs/block_64x64x64/ttshared.mlir \
  --df test/Dialect/DataflowDialect/2D_mesh.mlir \
  --dump-dir test/Passes/mm_2Dmesh/ \
  --skip-tile-scf-for-to-l1
``` -->
#### Option B step-by-step
1) Replace grid indices with a 3-D `affine.parallel`
```bash
build/tool/triton-shared/single_stage/grid_to_parallel \
  --input test/Passes/mm_2Dmesh/00_temp_manual_symbolic.mlir  \
  > test/Passes/mm_2Dmesh/01_after_grid_to_parallel.mlir
```

2) Enumerate spatial mappings and merge DF declarations
```bash
build/tool/triton-shared/single_stage/enumerate_hw_mapping \
  --input test/Passes/mm_2Dmesh/01_after_grid_to_parallel.mlir \
  --df test/Dialect/DataflowDialect/2D_mesh.mlir \
  > test/Passes/mm_2Dmesh/02_after_hardware_mapping.mlir
```

3) Hoist loading A, B blocks
```bash
build/tool/triton-shared/single_stage/hoist_block_loading \
  --input test/Passes/mm_2Dmesh/02_after_hardware_mapping.mlir \
  > test/Passes/mm_2Dmesh/03_after_block_hoisting.mlir
```

3) Annotate reuse on `memref.reinterpret_cast`
```bash
build/tool/triton-shared/single_stage/annotate_reuse \
  --input test/Passes/mm_2Dmesh/03_after_block_hoisting.mlir \
  > test/Passes/mm_2Dmesh/04_after_reuse_annotation.mlir
```

4) Explore alloc/copy mapping choices
```bash
build/tool/triton-shared/single_stage/explore_alloc_copy_mapping \
  --input test/Passes/mm_2Dmesh/04_after_reuse_annotation.mlir \
  > test/Passes/mm_2Dmesh/05_after_memref_mapping.mlir
```

5) Canonicalize 
```bash
build/tool/triton-shared/single_stage/canonicalize \
  --input test/Passes/mm_2Dmesh/05_after_memref_mapping.mlir \
  > test/Passes/mm_2Dmesh/06_after_canonicalization_.mlir
```
<!-- 4) Hoist block loading operations
**Note:** This pass does not have a standalone command-line tool. Use `mlir-opt` with the pass name:
```bash
build/tool/triton-shared/single_stage/hoist_block_loading \
  --input test/Passes/mm_2Dmesh/03_after_exploration.mlir \
  > test/Passes/mm_2Dmesh/04_after_hoist_block_loading.mlir
```



7) Bufferize tensors to memrefs
```bash
mlir-opt \
  --one-shot-bufferize="allow-unknown-ops allow-return-allocs-from-loops" \
  test/Passes/mm_2Dmesh/06_after_memref_mapping.mlir \
  > test/Passes/mm_2Dmesh/07_after_bufferization.mlir
```

8) Tile scf.for loops to fit L1 (optional)
```bash
build/tool/triton-shared/single_stage/tile_scf_for_to_l1 \
  test/Passes/mm_2Dmesh/07_after_bufferization.mlir \
  > test/Passes/mm_2Dmesh/08_after_for_tiling.mlir -->
``` 

Notes:
- The end-to-end driver accepts `--map-analysis-only` to attach `loom.copy.candidates` without cloning functions.
- The single-stage alloc/copy explorer accepts `--analysis-only` with the same effect.

### Performance Benchmarking
The `benchmark.sh` script measures command execution time with statistical analysis. It performs warmup runs to reduce cold-start effects, then runs multiple benchmark iterations and reports mean, median, min, max, and standard deviation (all in milliseconds).

```bash
# Basic usage
./benchmark.sh -- build/tool/ttshared-opt \
  --ttshared test/Triton/mm_fixed_strides/runs/block_64x64x64/ttshared.mlir \
  --df test/Dialect/DataflowDialect/2D_mesh.mlir \
  --dump-dir test/Passes/mm_2Dmesh/ \
  --skip-tile-scf-for-to-l1

# Customize warmup and benchmark runs
./benchmark.sh --warmup=5 --runs=20 -- build/tool/ttshared-opt ...

# Quiet mode (suppress command output)
./benchmark.sh -q -- build/tool/ttshared-opt ...
```

Options: `--warmup=N` (default: 3), `--runs=N` (default: 10), `-q/--quiet`, `-h/--help`.

### Pass reference (purpose, limitations, implementation)

- Constant deduplication and cleanup (`loom-const-cleanup`)
  - Purpose: Deduplicate `arith.constant` and `arith.constant_index` operations by value and type, remove unused constants, and fold constant operands into `affine.apply` operations to simplify IR.
  - Limitations: Only handles constants that are directly unused or can be folded into affine operations. Does not perform cross-function constant sharing.
  - Implementation: `lib/passes/triton-shared/const_dedup_cleanup.{h,cpp}`. This pass is automatically run after each major transformation in the pipeline.

- Affinize Triton-shared indices (`loom-triton-shared-affinize`)
  - Purpose: Rewrite arithmetic index expressions into `affine.apply`, convert eligible loads/stores to affine form, and express `memref.reinterpret_cast` offsets via affine maps. Treats trailing grid/thread arguments as dims/symbols to expose GPU-style indexing to affine. Promotes 32-bit integer ABI args to `index` type where needed.
  - Limitations: Conservative—only provably affine expressions are converted. Assumes the last 6 function arguments encode grid sizes/indices; nonconforming kernels are left unchanged. Some `memref` ops remain non-affine if indices are not proven affine. Signed division is not converted to affine (to avoid trunc-vs-floor mismatch).
  - Implementation: See `lib/passes/triton-shared/triton_shared_affinize.{h,cpp}`; pass argument is `loom-triton-shared-affinize`. CLI: `build/tool/triton-shared/single_stage/affinize`.

- Grid-to-parallel (`loom-triton-shared-grid-to-parallel`)
  - Purpose: Replace the last three grid index arguments with a 3-D `affine.parallel` with dynamic uppers `(sizeX,sizeY,sizeZ)`; erase index args from the signature and replace their uses by the parallel IVs. This makes the GPU launch grid explicit as a parallel loop structure.
  - Limitations: Requires ≥ 6 function args following `(sizeX,sizeY,sizeZ, idxX,idxY,idxZ)`; otherwise no-op. Expects sizes to be of `index` (affinization establishes this in typical flows). The resulting parallel has no reductions and yields no values.
  - Implementation: See `lib/passes/triton-shared/triton_shared_grid_to_parallel.{h,cpp}`. CLI: `build/tool/triton-shared/single_stage/grid_to_parallel`.

- Spatial mapping exploration (`loom-triton-shared-explore-spatial-mappings`)
  - Purpose: Enumerate mappings from hardware `df.spatial_dim` declarations to the outermost `affine.parallel` iterators; clone per mapping, annotate inner loops with `loom.mapped_to`, and insert outer `affine.for` "waves" when the mesh cannot cover the grid in one shot. Merges DF declarations into the module.
  - Limitations: Combinatorial growth in clones due to partitioning/permutation of dims and outer-for orderings. Exploration is structural (not resource-capacity aware) in this prototype. Only enumerates the first outermost `affine.parallel` per function.
  - Implementation: `lib/passes/triton-shared/enumerate_hw_mapping.{h,cpp}` (`EnumerateSpatialMappings`) and `triton_shared_spatial_mapping_pass.cpp`. CLI: `build/tool/triton-shared/single_stage/enumerate_hw_mapping`.

- Hoist block loading (`loom-hoist-block-loading`)
  - Purpose: Hoist block loading operations from innermost `scf.for` loops to outer loop levels. For each function, identifies loading blocks (patterns of operations that load data blocks), clones the function per loading block, and hoists each block to reduce redundant memory accesses.
  - Limitations: Only processes innermost `scf.for` loops that contain recognized loading block patterns. Functions without valid loading blocks are skipped. Block identification may miss non-standard patterns.
  - Implementation: `lib/passes/triton-shared/hoist_block_loading.{h,cpp}` and `block_loading_pattern.{h,cpp}`. CLI: `build/tool/triton-shared/single_stage/hoist_block_loading` (or `mlir-opt --loom-hoist-block-loading` or the end-to-end `ttshared-opt` driver).

- Reuse annotation on reinterpret-cast (`loom-annotate-reinterpret-cast-reuse`)
  - Purpose: Attach a `loom.reuse` dictionary to each `memref.reinterpret_cast` describing how its offset varies with surrounding iterators, grouped by `spatial` (`affine.parallel`), `temporal` (`affine.for`), and `sequential` (`scf.for`). Each entry records `iterator` (SSA name), `depth`, `reuse_type` (`no_reuse`/`total_reuse`), `volume` (bytes; 0, full block, or -1 unknown), and `mapped_to` for spatial entries.
  - Limitations: Binary reuse classification only (no partial reuse yet). Volumes require known block sizes. Dependency analysis is conservative and may miss some reuse opportunities.
  - Implementation: `lib/passes/triton-shared/reinterpret_cast_reuse.{h,cpp}`. CLI: `build/tool/triton-shared/single_stage/annotate_reuse`.

- Alloc/Copy mapping exploration (`loom-explore-alloc-copy-mapping`)
  - Purpose: Annotate `memref.alloc` with `{loom.alloc={local=true, memory_name=…}}` and enumerate per-`memref.copy` mapping choices: local memory copies and broadcasts along dimensions with spatial total-reuse. Merge the DF module to discover one `df.memory` and classify `df.interconnects` as x/y based on affine maps. Supports analysis-only mode via `--analysis-only` flag.
  - Limitations: Assumes a single `df.memory`; interconnect classification is heuristic (e.g., `(d0+1,d1)`→x, `(d0,d1+1)`→y). Enumerating the cross-product of candidates can explode; use analysis-only when needed. Requires prior reuse analysis to identify total-reuse dimensions.
  - Implementation: `lib/passes/triton-shared/explore_alloc_copy_mapping.{h,cpp}` and notes in `lib/passes/triton-shared/README.alloc_copy_mapping.md`. CLI: `build/tool/triton-shared/single_stage/explore_alloc_copy_mapping` (or end-to-end via `build/tool/ttshared-opt`).

- Tile scf.for loops to L1 (`loom-tile-scf-for-to-l1`)
  - Purpose: Tile `scf.for` loops so that per-tile memory fits within the single `df.memory` (L1) capacity. Computes per-iteration memory from `memref.alloc` operations annotated with `loom.alloc`, picks the largest power-of-two tile factor that fits, and rewrites loops into outer/inner tile structure.
  - Limitations: Requires fully bufferized IR (no tensor types). Assumes exactly one `df.memory`. Requires statically provable loop trip counts with exact divisibility by tile factor. Only considers allocs explicitly annotated as local to the single `df.memory`.
  - Implementation: `lib/passes/triton-shared/tile_scf_for_to_l1.{h,cpp}`. CLI: `build/tool/triton-shared/single_stage/tile_scf_for_to_l1`.

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
