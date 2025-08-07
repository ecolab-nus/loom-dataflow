# TMD

A C++ project template with CMake and GoogleTest integration.

## Dependencies

Before building this project, you need to install the following dependencies:

### On Arch Linux:
```bash
sudo pacman -S cmake gcc make
```

### On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install cmake build-essential
```

### On macOS:
```bash
brew install cmake
```

### On Windows:
- Install Visual Studio 2019 or later with C++ support
- Install CMake from https://cmake.org/download/

### MLIR Dependencies (Required):

This project requires MLIR support for the standalone dialect with specific default installation paths:

1. **Build and install LLVM/MLIR**: Follow the instructions in `src/scaleout/standalone/README.md` to build LLVM with MLIR support and install it to the **exact default location**: `$HOME/opt/llvm-mlir`

2. **LLVM-lit location**: The build expects `llvm-lit` to be available at: `$HOME/llvm-project/build/bin/llvm-lit`

3. **Default CMake paths expected**:
   - **MLIR cmake files**: `$HOME/opt/llvm-mlir/lib/cmake/mlir`
   - **LLVM External Lit**: `$HOME/llvm-project/build/bin/llvm-lit`

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
   -DCMAKE_INSTALL_PREFIX=$HOME/opt/llvm-mlir \
   -DLLVM_BUILD_UTILS=ON \
   -DLLVM_INSTALL_UTILS=ON

cmake --build . --target check-mlir
ninja install
```

This will install MLIR cmake files to `$HOME/opt/llvm-mlir/lib/cmake/mlir` and llvm-lit will be available at `$HOME/llvm-project/build/bin/llvm-lit`.

## Building the Project

### Using the Build Script (Linux/macOS):

**Basic build:**
```bash
./build.sh
```

**Build with custom MLIR paths:**
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
  -DMLIR_DIR=$HOME/opt/llvm-mlir/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$HOME/llvm-project/build/bin/llvm-lit \
  -DLLVM_USE_LINKER=lld
cmake --build . --config Release
ctest --output-on-failure
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
1. Create `.cpp` and `.h` files in the `src/` directory
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
   # Build with debug symbols and compile commands
   rm -rf build && mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Debug
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