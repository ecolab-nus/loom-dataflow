<img src="assets/loom-logo.svg" alt="Loom Logo" width="250">

**loom-dataflow** is a sub-module of the [loom](https://github.com/anthropics/loom) project. It provides an MLIR-backed compiler pipeline for exploring spatial hardware mappings and generating constraint models for dataflow accelerators. The pipeline lowers tensor-level kernels through a series of analysis and transformation passes, culminating in bufferized IR annotated with hardware mapping and memory allocation decisions.

## Repository Map

```
lib/
  analysis/         — Static memory analysis library (shared by passes and debug CLI)
  dataflow-dialect/ — TableGen + C++ for the df MLIR dialect
  loom-dialect/     — TableGen + C++ for the loom MLIR dialect
  modules/          — Hardware topology compositions (2D mesh, torus, ring chains)
  passes/
    common/         — Shared analysis utilities (hardware discovery, affine utils, etc.)
    loom-opt/       — Core transformation passes
    lcs/            — Loom Compute Schedule: staged ETG builder and constraint expressions
    tt-opt/         — Post-bufferization TT optimization passes
  pipeline/         — High-level C++ API and Python bindings (pybind11)
  resources/        — Primitive hardware resource models (SRAM banks, rings, chains)
tool/
  loom-opt/single_stage/ — Single-stage CLI drivers for each pipeline pass
  tt-opt/single_stage/   — CLI driver for tt-opt
  dataflow-dialect/      — Dataflow dialect utilities
  resource-system/       — Hardware resource demos
  loom-lsp-server/       — LSP server for IDE support
test/
  Passes/mm_2Dmesh/      — Primary regression test (matrix multiply on 2D mesh)
  Passes/flashattn_2Dmesh/
  Passes/mm_ibmring/
  Dialect/               — Dialect syntax and semantics tests
```

## Requirements

- CMake ≥ 3.20, Ninja, a C++17 compiler, and `lld` (or another linker if you override `LLVM_USE_LINKER`).
- An installed LLVM/MLIR build that exports CMake packages. The scripts default to `MLIR_DIR=/opt/llvm-mlir/lib/cmake/mlir`.
- `lit` or `llvm-lit` on `PATH` for CTest-driven MLIR tests (`pipx install lit` is the recommended route).

Quick install (Linux/Debian):
```bash
sudo apt install cmake build-essential ninja-build lld
```

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

## Core Pass Pipeline

All binaries live under `build/tool/` after a build. The full pipeline is exercised by `run_pipeline.sh` (the regression test).

| Step | Tool | Purpose |
|------|------|---------|
| 1 | `tensor_canonicalize` | Specialize `linalg` destination operands; fold redundant `tensor.extract_slice` |
| 2 | `memory_binding` | Bind physical memory to tensor ops via `loom.alloc` annotations |
| 3 | `enumerate_hw_mapping` | Enumerate spatial mappings from `df.spatial_dim` declarations to `affine.parallel` iterators |
| 4 | `analyze_reuse` | Annotate each `loom.subview` with `loom.reuse` (spatial / temporal / sequential dims) |
| 5 | `enumerate_copy_broadcast` | Enumerate per-copy broadcast choices; annotate `memref.alloc` with `loom.alloc` |
| 6 | `staged_etg` | Build Staged Execution Task Graph → JSON constraint model |
| 7 | `canonicalize` | Materialize symbolic block sizes into concrete constants; canonicalize IR |
| 8 | `one_shot_bufferize` | One-shot bufferization: tensor ops → memref ops |
| 9 | `tt-opt` | Convert zero-initialized linalg matmul ops to Loom ops and fold redundant zero fills |

**Not in the default pipeline (under active development):**
- `hoist_block_loading` — hoist block loading operations from innermost loops to outer loop levels

### Running the regression test
```bash
cd third_party/loom-dataflow
./run_pipeline.sh
```

### Step-by-step example (mqa_decode)

```bash
# Step 1
build/tool/loom-opt/single_stage/tensor_canonicalize \
  --input test/Passes/mqa_decode/IR/00_from_helion_frontend.mlir \
  > test/Passes/mqa_decode/IR/01_tensor_canonicalized.mlir

# Step 2
build/tool/loom-opt/single_stage/memory_binding \
  --input test/Passes/mqa_decode/IR/01_tensor_canonicalized.mlir \
  > test/Passes/mqa_decode/IR/02_explicit_memory_access.mlir

# Step 3
build/tool/loom-opt/single_stage/enumerate_hw_mapping \
  --input test/Passes/mqa_decode/IR/02_explicit_memory_access.mlir \
  --hw_spec /root/loom/third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir \
  > test/Passes/mqa_decode/IR/03_after_hardware_mapping.mlir

# Step 4
build/tool/loom-opt/single_stage/analyze_reuse \
  --input test/Passes/mqa_decode/IR/03_after_hardware_mapping.mlir \
  > test/Passes/mqa_decode/IR/04_after_reuse_analyzation.mlir

# Step 5
build/tool/loom-opt/single_stage/enumerate_copy_broadcast \
  --input test/Passes/mqa_decode/IR/04_after_reuse_analyzation.mlir \
  > test/Passes/mqa_decode/IR/05_after_enumerate_broadcast.mlir

# Step 6 — emits JSON constraint model
build/tool/loom-opt/single_stage/staged_etg \
  --input test/Passes/mqa_decode/IR/05_after_enumerate_broadcast.mlir \
  --hw_spec /root/loom/third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir \
  --output test/Passes/mqa_decode/constraint_space/staged_etg_dump.json

# Step 7
build/tool/loom-opt/single_stage/canonicalize \
  --input test/Passes/mqa_decode/IR/05_after_enumerate_broadcast.mlir \
  > test/Passes/mqa_decode/IR/06_after_canonicalize.mlir

# Step 8
build/tool/loom-opt/single_stage/one_shot_bufferize \
  --input test/Passes/mqa_decode/IR/06_after_canonicalize.mlir \
  > test/Passes/mqa_decode/IR/07_after_osb.mlir

# Step 9
build/tool/tt-opt/single_stage/tt-opt \
  --input test/Passes/mqa_decode/IR/07_after_osb.mlir \
  > test/Passes/mqa_decode/IR/08_tt-opt.mlir
```

## Pass Reference

### `tensor_canonicalize` (`loom-linalg-destination-specialization` + `loom-fold-redundant-extract-slice`)
- **Purpose**: Identify and fold redundant elementwise accumulation patterns into `linalg` output operands (e.g., `matmul(A,B,fill(0)) + add(iter_args)` → `matmul(A,B,iter_args)`). Then remove no-op `tensor.extract_slice` operations.
- **Implementation**: `lib/passes/loom-opt/src/linalg_destination_specialization_pass.cpp`, `fold_redundant_extract_slice_pass.cpp`

### `memory_binding` (`loom-memory-binding`)
- **Purpose**: Transform bufferization patterns to `loom` dialect operations that bind physical memory allocations to tensor semantics for downstream dataflow analysis.
- **Implementation**: `lib/passes/loom-opt/src/memory_binding_pass.cpp`

### `enumerate_hw_mapping` (`loom-triton-shared-explore-spatial-mappings`)
- **Purpose**: Enumerate all valid assignments of `df.spatial_dim` hardware dimensions to the outermost `affine.parallel` iterators. Clones the function per mapping, annotates inner loops with `loom.mapped_to`, and inserts outer `affine.for` wave loops when the mesh size does not cover the iteration space in one shot.
- **Implementation**: `lib/passes/loom-opt/src/triton_shared_spatial_mapping_pass.cpp`

### `analyze_reuse` (`loom-annotate-subview-reuse`)
- **Purpose**: Attach a `loom.reuse` dictionary to each `loom.subview` describing how its offset varies with surrounding iterators (spatial / temporal / sequential). Records `reuse_type` (no\_reuse / total\_reuse) and volume per dimension.
- **Implementation**: `lib/passes/loom-opt/src/analyze_reuse.cpp`

### `enumerate_copy_broadcast` (`loom-enumerate-copy-broadcast`)
- **Purpose**: For each `loom.copy_to_tensor`, enumerate whether the copy should use local memory or broadcast along dimensions with total spatial reuse. Annotates `memref.alloc` with `loom.alloc` carrying the candidate set. Supports `--analysis-only` mode to annotate without cloning.
- **Implementation**: `lib/passes/loom-opt/src/enumerate_copy_broadcast.cpp`

### `staged_etg`
- **Purpose**: Traverse the annotated IR and construct a Staged Execution Task Graph. Emits a JSON constraint model describing compute and communication schedules for use by the SMT solver in the broader loom pipeline.
- **Implementation**: `lib/passes/lcs/src/staged_etg_builder.cpp`; CLI driver: `tool/loom-opt/single_stage/staged_etg_main.cpp`

### `canonicalize` (`loom-materialize` + standard canonicalization)
- **Purpose**: Replace `loom.sym` symbolic block-size variables with concrete `arith.constant` values from the SMT solver result map. Variants for which no feasible solution exists are dropped with a diagnostic warning.
- **Implementation**: `lib/passes/loom-opt/src/materialize.cpp`

### `one_shot_bufferize`
- **Purpose**: Run MLIR's one-shot bufferization to lower tensor-level IR to memref-based IR in a single pass, using the `loom` dialect's custom bufferization interface.
- **Implementation**: `lib/loom-dialect/Transforms/BufferizableOpInterfaceImpl.h`; CLI driver: `tool/loom-opt/single_stage/one_shot_bufferize_main.cpp`

### `tt-opt` (`tt-convert-zero-fill-linalg-matmul-to-loom` + `tt-fold-zero-fill-linalg`)
- **Purpose**: Convert same-block zero-initialized `linalg.matmul` and `linalg.batch_matmul` ops to `loom.matmul` and `loom.batch_matmul`, then remove redundant zero `linalg.fill` ops feeding remaining destination-style `linalg` ops when there is no intervening use.
- **Implementation**: `lib/passes/tt-opt/src/convert_zero_fill_linalg_matmul_to_loom_pass.cpp`, `lib/passes/tt-opt/src/fold_zero_fill_linalg_pass.cpp`

### `hoist_block_loading` (`loom-hoist-block-loading`) *(under active development)*
- **Purpose**: Hoist block loading operations from innermost `affine.for` loops to outer loop levels to reduce redundant memory accesses. Identifies `loom.alloc + loom.copy_to_tensor` loading block patterns and clones the function per loading block.
- **Status**: Builds but is not part of the default pipeline. Updates pending.
- **Implementation**: `lib/passes/loom-opt/src/hoist_block_loading.cpp`, `lib/passes/common/src/block_loading_pattern.cpp`

## Debug Utilities

### `static_memory_analyser`
Standalone CLI for the memory analysis pass used internally by `memory_binding`. Parses an MLIR file and dumps the virtual buffer allocation plan (bucket grouping, coloring, liveness).

```bash
build/tool/loom-opt/single_stage/static_memory_analyser --input <file.mlir>
```

### Performance Benchmarking
`benchmark.sh` measures tool execution time with statistical analysis (mean, median, min, max, std dev across multiple runs).

```bash
./benchmark.sh --warmup=3 --runs=10 -- build/tool/loom-opt/single_stage/tensor_canonicalize \
  --input test/Passes/mqa_decode/IR/00_from_helion_frontend.mlir
```

## Tests

```bash
# Run all CTest-driven MLIR tests
cd build && ctest --output-on-failure

# Run the full end-to-end regression pipeline
cd third_party/loom-dataflow && ./run_pipeline.sh
```

Test cases are under `test/Passes/` (mm\_2Dmesh, flashattn\_2Dmesh, mm\_ibmring) and `test/Dialect/`.

## Troubleshooting

- `lit` not found: install via `pipx install lit` (preferred) or provide `--llvm-lit=/path/to/lit`.
- `MLIRConfig.cmake` missing: export `MLIR_DIR` to point at your LLVM/MLIR installation.
- IntelliSense gaps: rerun `./setup_ide.sh` so that `compile_commands.json` stays in sync with TableGen-generated headers.

## License

See `LICENSE`.
