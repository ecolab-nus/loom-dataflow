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

## Building the Project

### Using the Build Script (Linux/macOS):
```bash
./build.sh
```

### Manual Build:
```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build . --config Release

# Run tests
ctest --output-on-failure
```

## Project Structure

```
tmd/
├── CMakeLists.txt          # Main CMake configuration
├── build.sh               # Build script
├── src/                   # Source files
│   ├── main.cpp           # Main application entry point
│   ├── calculator.h       # Example header file
│   └── calculator.cpp     # Example implementation
└── tests/                 # Test files
    └── test_calculator.cpp # GoogleTest unit tests
```

## Running

After building, you can run:

- **Main application**: `./build/tmd`
- **Tests**: `./build/tmd_tests` or `cd build && ctest`

## Features

- **Modern CMake**: Uses CMake 3.14+ with modern practices
- **GoogleTest Integration**: Automatic download and setup using FetchContent
- **Cross-platform**: Works on Linux, macOS, and Windows
- **Automatic Test Discovery**: Tests are automatically discovered by CTest
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