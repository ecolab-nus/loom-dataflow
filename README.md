# TMD

An MLIR-based C++ project for exploring dataflow architectures, custom dialects, and compiler passes. It includes:
- A custom MLIR dialect `df` for hardware-aware data movement
- Static libraries for scale-out and scale-in concepts
- A small resource management framework and demos
- GoogleTest-based tests and IDE-friendly setup

## Repository layout

- `bin/` – entry points and tools
  - `main.cpp` → builds `tmd`
  - `resource_demo.cpp` → builds `tmd_resource_demo`
  - `module_demo.cpp` → builds `tmd_module_demo` (if present)
  - `df_parse_print.cpp` → builds `tmd_df_parse_print` (loads/prints MLIR with `df` dialect)
- `lib/scaleout/` – scale-out library and MLIR components
  - `dataflow-dialect/IR/` – `df` dialect TableGen and C++ (ops, types, dialect)
  - `analyses/`, `modules/`, `resources/` – helper libraries for modeling hardware
- `lib/scalein/` – scale-in library
- `tests/` – GoogleTest sources
- `build.sh` – Release build + test runner
- `setup_ide.sh` – Debug build and `compile_commands.json` generator

## Dependencies

Install toolchain and MLIR:

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install cmake build-essential ninja-build lld
```

### Arch Linux
```bash
sudo pacman -S cmake gcc make ninja lld
```

### macOS
```bash
brew install cmake ninja
```

### Windows
- Visual Studio 2019+ with C++
- CMake from `https://cmake.org/download/`

### MLIR (required)
- Provide an installed MLIR with CMake config files. Default expected path:
  - `MLIR_DIR=/opt/llvm-mlir/lib/cmake/mlir`
- Lit runner on PATH (prefer `lit`; `llvm-lit` also works)
  - `pipx install lit` then `pipx ensurepath` and open a new shell, or
  - `python3 -m venv ~/.venvs/lit && ~/.venvs/lit/bin/pip install lit`

You can override paths via `./build.sh --mlir-dir=… --llvm-lit=…` or CMake cache vars `-DMLIR_DIR=… -DLLVM_EXTERNAL_LIT=…`.

#### Quick reference: Build and install LLVM/MLIR
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

## Build

### One-liner (recommended)
```bash
./build.sh
```

Options:
```bash
./build.sh --mlir-dir=/custom/path/to/mlir --llvm-lit=/path/to/lit
./build.sh --help
```

### Manual build
```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/opt/llvm-mlir/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$(command -v lit || command -v llvm-lit) \
  -DLLVM_USE_LINKER=lld
cmake --build . --config Release
ctest --output-on-failure
```

### IDE setup (Debug + compile_commands.json)
```bash
./setup_ide.sh
```

## Run

- App: `./build/tmd`
- Resource demo: `./build/tmd_resource_demo`
- Module demo: `./build/tmd_module_demo` (if built)
- MLIR df tool: `./build/tmd_df_parse_print file.mlir`
- Tests: `./build/tmd_tests` or `cd build && ctest`

## The Dataflow (df) dialect

The `df` dialect models hardware-aware data movement and interconnects.
Key pieces live in `lib/scaleout/dataflow-dialect/IR/` and build the library `tmdDataflowDialect`.

- Example interconnect and load:
```mlir
// Declare 8x8 spatial grid and two affine interconnects
%x = df.spatial_dim 8
%y = df.spatial_dim 8
%horizontal_chains = "df.interconnects"(%x, %y) {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : (index, index) -> !df.interconnect
%vertical_chains   = "df.interconnects"(%x, %y) {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : (index, index) -> !df.interconnect

// Chained load over a vertical chain, using affine indices
%v = df.chained_load %mem[%i + 3, %j + 7] : memref<100x100xf32>, over %vertical_chains
```

See `lib/scaleout/dataflow-dialect/README.md` for more details.

## Testing

Tests are discovered via CTest. After any build:
```bash
cd build && ctest --output-on-failure
```

## Troubleshooting

- Lit not found: install `lit` as shown above or pass `--llvm-lit=/path/to/lit`.
- MLIR not found: set `MLIR_DIR` to your install, e.g. `-DMLIR_DIR=/opt/llvm-mlir/lib/cmake/mlir`.
- IntelliSense issues: run `./setup_ide.sh` and restart your IDE; ensure `build/compile_commands.json` exists.

## License

See `LICENSE`.