#!/bin/bash

# Build script for the TMD project with MLIR support

set -e

# Set default MLIR paths (can be overridden via environment variables)
MLIR_DIR=${MLIR_DIR:-$HOME/opt/llvm-mlir/lib/cmake/mlir}
LLVM_EXTERNAL_LIT=${LLVM_EXTERNAL_LIT:-$HOME/llvm-project/build/bin/llvm-lit}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mlir-dir=*)
      MLIR_DIR="${1#*=}"
      shift
      ;;
    --llvm-lit=*)
      LLVM_EXTERNAL_LIT="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --mlir-dir=PATH        Path to MLIR cmake files (default: \$HOME/opt/llvm-mlir/lib/cmake/mlir)"
      echo "  --llvm-lit=PATH        Path to llvm-lit executable (default: \$HOME/llvm-project/build/bin/llvm-lit)"
      echo "  -h, --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Create build directory
mkdir -p build
cd build

# Configure with CMake (MLIR is always enabled)
echo "Configuring project with CMake..."
echo "MLIR directory: $MLIR_DIR"
echo "LLVM Lit: $LLVM_EXTERNAL_LIT"

cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR=$MLIR_DIR \
    -DLLVM_EXTERNAL_LIT=$LLVM_EXTERNAL_LIT \
    -DLLVM_USE_LINKER=lld

# Build the project
echo "Building project..."
cmake --build . --config Release

# Run tests
echo "Running tests..."
ctest --output-on-failure

echo "Build completed successfully!"
echo ""
echo "Executables:"
echo "  - Main program: ./tmd"
echo "  - Resource management demo: ./tmd_resource_demo"
echo "  - Tests: ./tmd_tests"
echo ""
echo "MLIR Standalone Tools:"
echo "  - standalone-opt: ./standalone-opt"
echo "  - standalone-translate: ./standalone-translate"
echo ""
echo "To run MLIR tests:"
echo "  cmake --build . --target check-standalone"