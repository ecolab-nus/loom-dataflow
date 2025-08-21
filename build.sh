#!/bin/bash

# Build script for the TMD project with MLIR support

set -e

# Set default MLIR path (can be overridden via environment variables)
MLIR_DIR=${MLIR_DIR:-/opt/llvm-mlir/lib/cmake/mlir}
# Do not hardcode a default for LIT; auto-detect later unless provided
LLVM_EXTERNAL_LIT=${LLVM_EXTERNAL_LIT:-}

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
      echo "  --mlir-dir=PATH        Path to MLIR cmake files (default: /opt/llvm-mlir/lib/cmake/mlir)"
      echo "  --llvm-lit=PATH        Path to llvm-lit executable (default: auto-detect from PATH)"
      echo "  -h, --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Resolve repository root
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="$ROOT_DIR/build"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake (MLIR is always enabled)
echo "Configuring project with CMake..."
echo "MLIR directory: $MLIR_DIR"
echo "Build directory: $BUILD_DIR"

# Auto-detect lit/llvm-lit if not provided (prefer Python 'lit')
if [[ -z "$LLVM_EXTERNAL_LIT" ]]; then
  if command -v lit >/dev/null 2>&1; then
    LLVM_EXTERNAL_LIT=$(command -v lit)
  elif command -v llvm-lit >/dev/null 2>&1; then
    LLVM_EXTERNAL_LIT=$(command -v llvm-lit)
  elif [[ -x "$HOME/llvm-project/build/bin/llvm-lit" ]]; then
    LLVM_EXTERNAL_LIT="$HOME/llvm-project/build/bin/llvm-lit"
  else
    echo "ERROR: llvm-lit not found."
    echo "Install lit via one of:"
    echo "  - pipx install lit (preferred); then run: pipx ensurepath and open a new shell"
    echo "  - python3 -m venv ~/.venvs/lit && ~/.venvs/lit/bin/pip install lit"
    echo "Then ensure it is on your PATH, or pass --llvm-lit=/path/to/lit"
    exit 1
  fi
fi

echo "LLVM External Lit: $LLVM_EXTERNAL_LIT"

cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR="$MLIR_DIR" \
    -DLLVM_EXTERNAL_LIT="$LLVM_EXTERNAL_LIT" \
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