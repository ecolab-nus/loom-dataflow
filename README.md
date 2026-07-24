<img src="assets/loom-logo.svg" alt="Loom Logo" width="250">

**loom-dataflow** is a sub-module of the [loom](https://github.com/anthropics/loom) project. It provides an MLIR-backed compiler pipeline for exploring spatial hardware mappings and generating constraint models for dataflow accelerators. The pipeline lowers tensor-level kernels through analysis and transformation passes, then materializes selected block sizes into bufferized IR with Loom dialect operations.

## Repository Map

```
lib/
  adl-dialect/      - TableGen + C++ for the ADL hardware-description dialect
  analysis/         - Static memory analysis and hardware-dimension splitting helpers
  loom-dialect/     - TableGen + C++ for the loom MLIR dialect and bufferization hooks
  modules/          - Hardware topology compositions (2D mesh, torus, ring chains)
  passes/
    common/         - Shared utilities for hardware discovery, mapping, affine helpers, and drivers
    loom-opt/       - Core Loom transformation and exploration passes
    lcs/            - Loom Compute Schedule: staged ETG builder and constraint expressions
    tt-opt/         - TT-oriented post-bufferization optimization passes
  pipeline/         - High-level C++ API and Python bindings (pybind11)
  resources/        - Primitive hardware resource models (memory, rings, chains)
tool/
  loom-opt/             - `loom-opt` plus single-stage CLI drivers for each pipeline pass
  tt-opt/single_stage/  - CLI driver for TT cleanup passes
  adl-dialect/          - ADL parser utility
  resource-system/      - Hardware resource demos
  loom-lsp-server/      - LSP server for IDE support
test/
  Passes/           - Saved pipeline inputs/outputs for kernels such as mm, mqa_decode, flashattn
  Triton/           - Triton source and IR captures used to create pipeline inputs
  Dialect/Affine/   - Affine analysis examples
```

## Requirements

- CMake >= 3.20, Ninja, a C++17 compiler, and `lld` (or another linker if you override `LLVM_USE_LINKER`).
- An installed LLVM/MLIR build that exports CMake packages. The scripts default to `MLIR_DIR=/opt/llvm-mlir/lib/cmake/mlir`.
- Python >= 3.10 with development headers.
- `pybind11` and `scikit-build-core` for Python bindings and package builds.
- `lit` or `llvm-lit` on `PATH`; the build scripts require one and accept `--llvm-lit=/path/to/lit`.

Quick install (Linux/Debian):
```bash
sudo apt install cmake build-essential ninja-build lld python3-dev
python3 -m pip install pybind11 scikit-build-core lit
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

### Python package build
```bash
python3 -m pip install -e . -v --no-build-isolation
```

The Python package installs `loom_pipeline`, which wraps the C++ pipeline through pybind11. CMake can fetch pybind11 if it cannot find a local install, but a local Python package is preferred for offline or restricted environments.

### IDE/Debug setup
`./setup_ide.sh` performs a clean Debug build and emits `build/compile_commands.json` for IntelliSense.

## Core Pass Pipeline

All binaries live under `build/tool/` after a build. The scripted file-based pipeline is:

```bash
cd third_party/loom-dataflow
./run_pipeline.sh mqa_decode
./run_pipeline.sh mqa_decode "[1,2,3]"
```

The first argument is the test case under `test/Passes/`. The optional second argument selects which numbered steps to run.

| Step | Tool | Purpose |
|------|------|---------|
| 1 | `tensor_canonicalize` | Normalize tensor IR, specialize linalg destinations, insert handoff helpers, and prepare bufferization anchors |
| 2 | `memory_binding` | Bind physical memory to tensor-level operations with Loom dialect handoff/copy operations |
| 3 | `enumerate_hw_mapping` | Enumerate mappings from ADL hardware dimensions to `affine.parallel` iterators |
| 4 | `analyze_reuse` | Annotate each `loom.subview` with spatial, temporal, and sequential reuse information |
| 5 | `enumerate_copy_broadcast` | Enumerate local-copy and broadcast candidates for copy-to-tensor operations |
| 6 | `staged_etg` | Build a Staged Execution Task Graph and emit JSON constraints |
| 7 | `canonicalize` | Materialize selected block-size values and canonicalize the IR |
| 8 | `one_shot_bufferize` | Bridge Loom view ops and run MLIR one-shot bufferization |
| 9 | `tt-opt` | Run TT cleanup: Loom matmul conversion, zero-fill folding, and scalar-chain splitting |

`hoist_block_loading` also builds as a standalone tool, but it is not part of `run_pipeline.sh`.

### Standalone pass tools

These tools read MLIR with `--input`; tools that need hardware use `--hw_spec`, and `staged_etg` writes JSON with `--output`.

| Tool | Example | Description |
|------|---------|-------------|
| `build/tool/loom-opt/single_stage/tensor_canonicalize` | `--input in.mlir > out.mlir` | Runs guarded linalg fusion, destination specialization, extract-slice folding, and bufferization handoff preparation. |
| `build/tool/loom-opt/single_stage/memory_binding` | `--input in.mlir > out.mlir` | Rewrites tensor/memory handoffs into Loom dialect memory operations. |
| `build/tool/loom-opt/single_stage/enumerate_hw_mapping` | `--input in.mlir --hw_spec arch.mlir > out.mlir` | Loads an ADL hardware spec, clones mapping candidates, and merges hardware declarations into the output module. |
| `build/tool/loom-opt/single_stage/hoist_block_loading` | `--input in.mlir > out.mlir` | Hoists recognized block-loading patterns for experimentation outside the default pipeline. |
| `build/tool/loom-opt/single_stage/analyze_reuse` | `--input in.mlir > out.mlir` | Adds reuse metadata to Loom subviews based on surrounding loop iterators. |
| `build/tool/loom-opt/single_stage/enumerate_copy_broadcast` | `--input in.mlir > out.mlir` | Enumerates copy placement and broadcast choices from reuse metadata. |
| `build/tool/loom-opt/single_stage/staged_etg` | `--input in.mlir --hw_spec arch.mlir --output etg.json` | Emits the constraint JSON consumed by the external block-size solver. |
| `build/tool/loom-opt/single_stage/canonicalize` | `--input in.mlir > out.mlir` | Runs Loom materialization and canonical cleanup. |
| `build/tool/loom-opt/single_stage/one_shot_bufferize` | `--input in.mlir > out.mlir` | Converts tensor IR to memref IR using Loom bufferization support. |
| `build/tool/loom-opt/single_stage/static_memory_analyser` | `--input in.mlir` | Dumps the memory analysis plan used by `memory_binding`. |
| `build/tool/tt-opt/single_stage/tt-opt` | `--input in.mlir > out.mlir` | Applies TT-oriented Loom/linalg cleanup passes after bufferization. |

### Python API

The in-memory API avoids intermediate files:

```python
from loom_pipeline import run_exploration, run_materialization

explored_mlir, etg_json = run_exploration(input_mlir, "path/to/arch.mlir")
final_mlir = run_materialization(explored_mlir, block_sizes_json)
```

`run_exploration` covers tensor canonicalization through copy-broadcast enumeration and can emit ETG JSON. `run_materialization` applies block-size bindings, bufferization, and TT cleanup.

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
# Changing the knob full_occ to true skips occupancy enumerating
build/tool/loom-opt/single_stage/enumerate_hw_mapping \
  --full_occ=false \
  --input test/Passes/mqa_decode/IR/02_explicit_memory_access.mlir \
  --hw_spec ../loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir \
  > test/Passes/mqa_decode/IR/03_after_hardware_mapping.mlir

# Step 4
build/tool/loom-opt/single_stage/analyze_reuse \
  --input test/Passes/mqa_decode/IR/03_after_hardware_mapping.mlir \
  > test/Passes/mqa_decode/IR/04_after_reuse_analyzation.mlir

# Step 5
build/tool/loom-opt/single_stage/enumerate_copy_broadcast \
  --input test/Passes/mqa_decode/IR/04_after_reuse_analyzation.mlir \
  > test/Passes/mqa_decode/IR/05_after_enumerate_broadcast.mlir

# Step 6
build/tool/loom-opt/single_stage/staged_etg \
  --input test/Passes/mqa_decode/IR/05_after_enumerate_broadcast.mlir \
  --hw_spec ../loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir \
  --output test/Passes/mqa_decode/constraint_space/staged_etg_dump.json

# Step 7
build/tool/loom-opt/single_stage/canonicalize \
  --input test/Passes/mqa_decode/IR/05_after_enumerate_broadcast.mlir \
  > test/Passes/mqa_decode/IR/06_after_canonicalize.mlir

# Step 8
build/tool/loom-opt/single_stage/one_shot_bufferize \
  --input test/Passes/mqa_decode/IR/06_after_canonicalize.mlir \
  > test/Passes/mqa_decode/IR/07_after_osb.mlir

# Step 9 (Hardware specific post-processing pass, Optional)
build/tool/tt-opt/single_stage/tt-opt \
  --input test/Passes/mqa_decode/IR/07_after_osb.mlir \
  > test/Passes/mqa_decode/IR/08_tt-opt.mlir
```

## Pass Reference

### `tensor_canonicalize`
- **Purpose**: Run the front of the Loom tensor pipeline: guarded linalg elementwise fusion, linalg destination specialization, redundant extract-slice folding, preparation-op sinking, loop handoff proxy insertion, and canonical bufferization-to-Loom rewrites.
- **Implementation**: `lib/passes/loom-opt/src/linalg_guarded_elementwise_fusion_pass.cpp`, `linalg_destination_specialization_pass.cpp`, `fold_redundant_extract_slice_pass.cpp`, `sink_preparation_ops_pass.cpp`, `loop_handoff_proxy_copy_insertion_pass.cpp`, `canonical_bufferization_to_loom_pass.cpp`

### `memory_binding` (`loom-memory-binding`)
- **Purpose**: Transform tensor-level bufferization patterns to Loom dialect operations that explicitly bind physical memory allocations for downstream dataflow analysis.
- **Implementation**: `lib/passes/loom-opt/src/memory_binding_pass.cpp`

### `enumerate_hw_mapping` (`loom-triton-shared-explore-spatial-mappings`)
- **Purpose**: Enumerate assignments from ADL hardware dimensions to outer `affine.parallel` iterators, clone functions per candidate, annotate mapped loops, and insert wave loops when needed.
- **Implementation**: `lib/passes/loom-opt/src/triton_shared_spatial_mapping_pass.cpp`, `lib/passes/common/src/hardware_mapping.cpp`

### `analyze_reuse` (`loom-annotate-subview-reuse`)
- **Purpose**: Attach reuse metadata to each `loom.subview`, grouped by spatial, temporal, and sequential iterators.
- **Implementation**: `lib/passes/loom-opt/src/analyze_reuse.cpp`

### `enumerate_copy_broadcast` (`loom-enumerate-copy-broadcast`)
- **Purpose**: Enumerate local-memory and broadcast choices for copies using reuse metadata, then annotate allocation candidates.
- **Implementation**: `lib/passes/loom-opt/src/enumerate_copy_broadcast.cpp`

### `staged_etg`
- **Purpose**: Traverse annotated IR and construct a Staged Execution Task Graph JSON constraint model.
- **Implementation**: `lib/passes/lcs/src/staged_etg_builder.cpp`; CLI driver: `tool/loom-opt/single_stage/staged_etg_main.cpp`

### `canonicalize` (`loom-materialize` + cleanup)
- **Purpose**: Replace `loom.get_module_attribute` operations with concrete values from selected block-size bindings and run canonical cleanup.
- **Implementation**: `lib/passes/loom-opt/src/materialize.cpp`

### `one_shot_bufferize`
- **Purpose**: Bridge Loom subviews to OSB-compatible memref operations, lower affine with Loom attributes preserved, and run MLIR one-shot bufferization.
- **Implementation**: `lib/passes/loom-opt/src/view_to_reinterpret_cast.cpp`, `lower_affine_with_attr.cpp`, `lower_linalg_copy_to_loom_copy_pass.cpp`, `lib/loom-dialect/Transforms/bufferizable_op_interface_impl.cpp`

### `tt-opt`
- **Purpose**: Convert zero-initialized `linalg.matmul` and `linalg.batch_matmul` ops to Loom ops, fold redundant zero fills, and split safe fused binary scalar chains.
- **Implementation**: `lib/passes/tt-opt/src/convert_zero_fill_linalg_matmul_to_loom_pass.cpp`, `fold_zero_fill_linalg_pass.cpp`, `split_binary_scalar_chain_pass.cpp`

### `hoist_block_loading` (`loom-hoist-block-loading`)
- **Purpose**: Hoist recognized block-loading operations from inner loops to outer loop levels for standalone experimentation.
- **Implementation**: `lib/passes/loom-opt/src/hoist_block_loading.cpp`, `lib/passes/common/src/block_loading_pattern.cpp`

## Debug Utilities

### `static_memory_analyser`
Standalone CLI for the memory analysis pass used internally by `memory_binding`. It parses an MLIR file and dumps the virtual buffer allocation plan.

```bash
build/tool/loom-opt/single_stage/static_memory_analyser --input <file.mlir>
```

### Performance Benchmarking
`benchmark.sh` measures tool execution time with statistical analysis.

```bash
./benchmark.sh --warmup=3 --runs=10 -- build/tool/loom-opt/single_stage/tensor_canonicalize \
  --input test/Passes/mqa_decode/IR/00_from_helion_frontend.mlir
```

## Troubleshooting

- `lit` not found: install via `python3 -m pip install lit` or provide `--llvm-lit=/path/to/lit`.
- `MLIRConfig.cmake` missing: export `MLIR_DIR` to point at your LLVM/MLIR installation.
- `pybind11` not found in a restricted environment: install it into the active Python environment before configuring.
- IntelliSense gaps: rerun `./setup_ide.sh` so that `compile_commands.json` stays in sync with TableGen-generated headers.

## License

See `LICENSE`.
