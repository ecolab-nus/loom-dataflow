# TMD

A C++ project template with CMake and GoogleTest integration.

## Dependencies

Before building this project, you need to install the following dependencies:

### On Arch Linux:
```bash
sudo pacman -S cmake gcc make ninja lld
```

### On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install cmake build-essential ninja-build lld
```

### On macOS:
```bash
brew install cmake ninja
```

### On Windows:
- Install Visual Studio 2019 or later with C++ support
- Install CMake from https://cmake.org/download/

### MLIR Dependencies (Required):

This project requires MLIR support for the standalone dialect with specific default installation paths:

1. **Build and install LLVM/MLIR**: Follow the instructions in `lib/scaleout/standalone/README.md` to build LLVM with MLIR support and install it to the **default location**: `/opt/llvm-mlir` (you can use another location, but then pass `-DMLIR_DIR=...` during configure or set it in your IDE settings)

2. **lit/llvm-lit location**: The build expects a lit runner on your PATH (prefer `lit`). Recommended installs:
   - `pipx install lit` (isolated, recommended). After install, run `pipx ensurepath` and open a new shell so `lit` is on PATH.
   - `python3 -m venv ~/.venvs/lit && ~/.venvs/lit/bin/pip install lit`

3. **Default CMake paths expected**:
   - **MLIR cmake files**: `/opt/llvm-mlir/lib/cmake/mlir`
    - **LLVM External Lit**: auto-detected (`which llvm-lit` or `which lit`)

4. **Additional dependencies**: 
   - Ninja build system: `sudo pacman -S ninja` (Arch), `sudo apt install ninja-build` (Ubuntu)
   - LLD linker (usually included with LLVM)

**Note**: If you have MLIR installed in different locations, you can override these paths using the build script options or CMake variables.

#### Quick Reference: Building LLVM/MLIR with Correct Paths

```bash
# Clone LLVM project (if not already done)
git clone https://github.com/llvm/llvm-project.git $HOME/llvm-project

# Build and install LLVM/MLIR to the expected location
cd $HOME/llvm-project
mkdir build && cd build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON \
   -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
   -DCMAKE_INSTALL_PREFIX=/opt/llvm-mlir \
   -DLLVM_BUILD_UTILS=ON \
   -DLLVM_INSTALL_UTILS=ON

cmake --build . --target check-mlir
ninja install
```

This will install MLIR cmake files to `/opt/llvm-mlir/lib/cmake/mlir`. Note that many installs do not ship `llvm-lit` in the install tree; prefer installing `lit` via pipx or in a venv and using it from your PATH.

## Building the Project

### Using the Build Script (Linux/macOS):

**Basic build:**
```bash
./build.sh
```

**Build with custom MLIR/Lit paths:**
```bash
./build.sh --mlir-dir=/custom/path/to/mlir --llvm-lit=/path/to/llvm-lit
```

**Get help:**
```bash
./build.sh --help
```

### Manual Build:

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

### Which script should I use?

- **build.sh**: One-shot configure + build + test in Release mode.
  - **Use when**: You want a production-like build and to run tests (CI/local sanity build).
  - **Behavior**: Creates `build/` if missing (no clean), auto-detects `lit/llvm-lit`, uses Ninja and `lld`.
  - **Override paths**: `./build.sh --mlir-dir=… --llvm-lit=…` or set `MLIR_DIR`/`LLVM_EXTERNAL_LIT` env vars.

- **setup_ide.sh**: Developer/IDE-oriented Debug build that generates `build/compile_commands.json`.
  - **Use when**: First-time IDE setup or when IntelliSense/symbol resolution is incorrect.
  - **Behavior**: Deletes and recreates `build/`, configures Debug with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, auto-detects `lit/llvm-lit`, builds, and verifies `compile_commands.json` exists.
  - **Note**: It does not write IDE settings files; IDEs (Cursor/VSCode) use `compile_commands.json` with the recommended extensions.

### Build directory behavior

- Both scripts use the same out-of-source directory: `build/`.
- `setup_ide.sh` runs a clean Debug configuration (`rm -rf build`), while `build.sh` performs a faster incremental Release build (`mkdir -p build`).
- Running them sequentially is fine; do not run them concurrently.
- To keep Debug and Release builds side-by-side, use separate directories with the manual commands, for example:

```bash
# Debug tree
cmake -S . -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DMLIR_DIR=/opt/llvm-mlir/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$(command -v lit || command -v llvm-lit) \
  -DLLVM_USE_LINKER=lld
cmake --build build-debug --parallel

# Release tree
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/opt/llvm-mlir/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$(command -v lit || command -v llvm-lit) \
  -DLLVM_USE_LINKER=lld
cmake --build build-release --config Release
```

## Running

After building, you can run:

- **Main application**: `./build/tmd`
- **Tests**: `./build/tmd_tests` or `cd build && ctest`
- **MLIR Standalone Tools**:
  - `./build/standalone-opt`: MLIR optimization tool
  - `./build/standalone-translate`: MLIR translation tool
- **MLIR Tests**: `cd build && cmake --build . --target check-standalone`

## Features

- **Modern CMake**: Uses CMake 3.20+ with modern practices
- **GoogleTest Integration**: Automatic download and setup using FetchContent
- **Cross-platform**: Works on Linux, macOS, and Windows  
- **Automatic Test Discovery**: Tests are automatically discovered by CTest
- **MLIR Integration**: Built-in MLIR standalone dialect support for advanced compiler tooling
- **Flexible Build Configuration**: Easy configuration for different MLIR paths
- **C++17 Standard**: Modern C++ features enabled
- **Compiler Warnings**: Strict warning levels enabled

## Adding New Code

### Adding Source Files:
1. Create `.cpp` and `.h` files in the `lib/` directory
2. CMake will automatically find and include them

### Adding Tests:
1. Create test files in the `tests/` directory with names like `test_*.cpp`
2. Use GoogleTest/GMock syntax for writing tests
3. CMake will automatically build and register them with CTest

## Example Test

```cpp
#include <gtest/gtest.h>
#include "your_class.h"

TEST(YourClassTest, BasicTest) {
    YourClass obj;
    EXPECT_EQ(obj.getValue(), 42);
}
```

## IDE Configuration

### Setting up C++ IntelliSense (Cursor/VSCode)

To resolve undefined symbols and get proper IntelliSense support:

1. **Quick Setup** (Recommended):
   ```bash
   ./setup_ide.sh
   ```

2. **Manual Setup**:
   ```bash
# Build with debug symbols and compile commands (Ninja generator)
rm -rf build && mkdir build && cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Debug -DMLIR_DIR=/opt/llvm-mlir/lib/cmake/mlir
   cmake --build . --parallel
   ```

3. **Restart your IDE** after running the setup to pick up the new configuration.

### What's Configured

- ✅ **compile_commands.json**: Generated for accurate symbol resolution
- ✅ **Include paths**: Automatic detection of project and GoogleTest headers
- ✅ **IntelliSense**: C++17 standard with proper compiler settings
- ✅ **Build tasks**: Integrated CMake build commands
- ✅ **Debug support**: GDB integration for debugging
- ✅ **Test runner**: Direct test execution from IDE

### VSCode Extensions Recommended

- **C/C++** (Microsoft) - Core language support
- **CMake Tools** - CMake integration
- **CMake** - CMake language support

### Troubleshooting

If you still see undefined symbols:
1. Run `./setup_ide.sh` to regenerate configuration
2. Restart Cursor/VSCode completely
3. Use `Ctrl+Shift+P` → `C/C++: Reload IntelliSense`
4. Check that `build/compile_commands.json` exists
5. Use `CMake: Delete Cache and Reconfigure` after switching generators or MLIR paths
6. If CTest cannot find the test runner, install `lit` and/or set: `-DLLVM_EXTERNAL_LIT=$(which llvm-lit || which lit)`

## CMake Options

You can customize the build by passing options to CMake:

```bash
# Debug build (recommended for development)
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Release build (default)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Specify compiler
cmake .. -DCMAKE_CXX_COMPILER=clang++

# Generate compile commands (for IDE support)
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```