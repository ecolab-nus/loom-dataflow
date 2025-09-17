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
- `triton_shared_affinize` – normalize a Triton-shared kernel to affine-friendly form.
- `triton_shared_grid_to_parallel` – wrap kernels in 3-D `affine.parallel` loops and drop explicit grid indices.
- `triton_shared_to_affine` – full pipeline that merges the transformed kernel with a DF module.
- `triton_shared_explore` – enumerate spatial mappings for Triton-shared kernels.
- `affine_explore`, `affine_tile`, `affine_analyze` – affine-only exploration, tiling, and reuse analysis utilities.

Example:
```bash
build/tool/triton_shared_to_affine \
  --ttshared test/Dialect/Triton/mm_normal/ttshared.mlir \
  --df test/Dialect/DataflowDialect/2D_mesh.mlir > merged.mlir
```

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
