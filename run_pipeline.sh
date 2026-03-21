#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Default values
CONSTRAINT_LINEARIZATION=true

echo "Starting build..."
if ! ./build.sh; then
    echo "Error: ./build.sh failed."
    exit 1
fi

echo "Creating necessary directories..."
mkdir -p test/Passes/mm_2Dmesh/IR
mkdir -p test/Passes/mm_2Dmesh/constraint_space
mkdir -p test/Passes/mm_2Dmesh/viz

echo "1) Specialize linalg operations' destination..."
if ! build/tool/loom-opt/single_stage/tensor_canonicalize \
  --input test/Passes/mm_2Dmesh/IR/00_from_helion_frontend.mlir  \
  > test/Passes/mm_2Dmesh/IR/01_tensor_canonicalized.mlir; then
    echo "Error: Step 1 failed."
    exit 1
fi

echo "2) Replace grid indices with a 3-D affine.parallel..."
if ! build/tool/loom-opt/single_stage/memory_binding \
  --input test/Passes/mm_2Dmesh/IR/01_tensor_canonicalized.mlir  \
  > test/Passes/mm_2Dmesh/IR/02_explicit_memory_access.mlir; then
    echo "Error: Step 2 failed."
    exit 1
fi

echo "3) Enumerate spatial mappings and merge DF declarations..."
if ! build/tool/loom-opt/single_stage/enumerate_hw_mapping \
  --input test/Passes/mm_2Dmesh/IR/02_explicit_memory_access.mlir \
  --df test/Dialect/DataflowDialect/2D_mesh.mlir \
  > test/Passes/mm_2Dmesh/IR/03_after_hardware_mapping.mlir; then
    echo "Error: Step 3 failed."
    exit 1
fi

# echo "3) Hoist loading A, B blocks..."
# if ! build/tool/loom-opt/single_stage/hoist_block_loading \
#   --input test/Passes/mm_2Dmesh/IR/02_after_hardware_mapping.mlir \
#   > test/Passes/mm_2Dmesh/IR/03_after_block_hoisting.mlir; then
#     echo "Error: Step 3 failed."
#     exit 1
# fi

echo "4) Analyze reuse pattern on loom.subview..."
if ! build/tool/loom-opt/single_stage/analyze_reuse \
  --input test/Passes/mm_2Dmesh/IR/03_after_hardware_mapping.mlir \
  > test/Passes/mm_2Dmesh/IR/04_after_reuse_analyzation.mlir; then
    echo "Error: Step 4 failed."
    exit 1
fi

echo "5) Enumerate copy interconnect broadcast choices on loom.copy_to_tensor..."
if ! build/tool/loom-opt/single_stage/enumerate_copy_broadcast \
  --input test/Passes/mm_2Dmesh/IR/04_after_reuse_analyzation.mlir \
  > test/Passes/mm_2Dmesh/IR/05_after_enumerate_broadcast.mlir; then
    echo "Error: Step 5 failed."
    exit 1
fi

echo "Dump ETG..."
if ! build/tool/loom-opt/single_stage/staged_etg \
  --input test/Passes/mm_2Dmesh/IR/05_after_enumerate_broadcast.mlir \
  --hw-compute-dir ../loom-mlar/tests/2d_mesh/compute \
  --output test/Passes/mm_2Dmesh/constraint_space/staged_etg_dump.json; then
    echo "Error: Dump ETG failed."
    exit 1
fi

# optional) Materialize symbolic block sizes
echo "6) Materialize symbolic block sizes..."
if ! build/tool/loom-opt/single_stage/canonicalize \
  --input test/Passes/mm_2Dmesh/IR/05_after_enumerate_broadcast.mlir \
  > test/Passes/mm_2Dmesh/IR/06_after_canonicalize.mlir; then
    echo "Error: Step 6 failed."
    exit 1
fi

echo "7) OSB..."
if ! build/tool/loom-opt/single_stage/one_shot_bufferize \
  --input test/Passes/mm_2Dmesh/IR/06_after_canonicalize.mlir \
  > test/Passes/mm_2Dmesh/IR/07_after_osb.mlir; then
    echo "Error: Step 7 failed."
    exit 1
fi

echo "8) tt-opt..."
if ! build/tool/tt-opt/single_stage/fuse_fill_matmul \
  --input test/Passes/mm_2Dmesh/IR/07_after_osb.mlir \
  > test/Passes/mm_2Dmesh/IR/08_tt-opt.mlir; then
    echo "Error: Step 8 failed."
    exit 1
fi

echo "All steps completed successfully."
