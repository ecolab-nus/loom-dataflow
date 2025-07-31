#!/bin/bash

# Build script for the TMD project

set -e

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring project with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

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