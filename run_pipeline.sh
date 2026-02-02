#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Default values
CONSTRAINT_LINEARIZATION=true

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --constraint_linearization|-cl) CONSTRAINT_LINEARIZATION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Starting build..."
if ! ./build.sh; then
    echo "Error: ./build.sh failed."
    exit 1
fi

echo "Creating necessary directories..."
mkdir -p test/Passes/mm_2Dmesh/IR
mkdir -p test/Passes/mm_2Dmesh/constraint_space
mkdir -p test/Passes/mm_2Dmesh/viz

echo "1) Replace grid indices with a 3-D affine.parallel..."
if ! build/tool/loom-opt/single_stage/memory_binding \
  --input test/Passes/mm_2Dmesh/IR/00_from_helion_frontend.mlir  \
  > test/Passes/mm_2Dmesh/IR/01_explicit_memory_access.mlir; then
    echo "Error: Step 1 failed."
    exit 1
fi

echo "2) Enumerate spatial mappings and merge DF declarations..."
if ! build/tool/loom-opt/single_stage/enumerate_hw_mapping \
  --input test/Passes/mm_2Dmesh/IR/01_explicit_memory_access.mlir \
  --df test/Dialect/DataflowDialect/2D_mesh.mlir \
  > test/Passes/mm_2Dmesh/IR/02_after_hardware_mapping.mlir; then
    echo "Error: Step 2 failed."
    exit 1
fi

# echo "3) Hoist loading A, B blocks..."
# if ! build/tool/loom-opt/single_stage/hoist_block_loading \
#   --input test/Passes/mm_2Dmesh/IR/02_after_hardware_mapping.mlir \
#   > test/Passes/mm_2Dmesh/IR/03_after_block_hoisting.mlir; then
#     echo "Error: Step 3 failed."
#     exit 1
# fi

echo "4) Analyze reuse pattern on loom.reinterpret_cast..."
if ! build/tool/loom-opt/single_stage/analyze_reuse \
  --input test/Passes/mm_2Dmesh/IR/02_after_hardware_mapping.mlir \
  > test/Passes/mm_2Dmesh/IR/04_after_reuse_analyzation.mlir; then
    echo "Error: Step 4 failed."
    exit 1
fi

echo "5) Enumerate copy interconnect broadcast choices on loom.copy..."
if ! build/tool/loom-opt/single_stage/enumerate_copy_broadcast \
  --input test/Passes/mm_2Dmesh/IR/04_after_reuse_analyzation.mlir \
  > test/Passes/mm_2Dmesh/IR/05_after_enumerate_broadcast.mlir \
  2> test/Passes/mm_2Dmesh/constraint_space/raw_constraint_space.json; then
    echo "Error: Step 5 failed."
    exit 1
fi

# optional) Materialize symbolic block sizes
echo "(optional) Materialize symbolic block sizes..."
if ! build/tool/loom-opt/single_stage/materialize \
  --input test/Passes/mm_2Dmesh/IR/05_after_enumerate_broadcast.mlir \
  > test/Passes/mm_2Dmesh/IR/06_after_materialize.mlir; then
    echo "Error: (optional) Materialize symbolic block sizes failed."
    exit 1
fi

if [ "$CONSTRAINT_LINEARIZATION" = "false" ]; then
    echo "Constraint linearization is disabled. Terminating pipeline after Step 5."
    exit 0
fi

echo "6) Canonicalize constraints..."
if ! ./build/tool/loom-constraint/single_stage/constraint_canonicalize \
  --input test/Passes/mm_2Dmesh/IR/05_after_enumerate_broadcast.mlir \
  > test/Passes/mm_2Dmesh/IR/11_after_canonicalize.mlir; then
    echo "Error: Step 6 failed."
    exit 1
fi

echo "7) Factorize constraints..."
if ! ./build/tool/loom-constraint/single_stage/constraint_factorize \
  --input test/Passes/mm_2Dmesh/IR/11_after_canonicalize.mlir \
  > test/Passes/mm_2Dmesh/IR/12_after_factorize.mlir; then
    echo "Error: Step 7 failed."
    exit 1
fi

echo "8) Decompose non-linear constraints..."
if ! ./build/tool/loom-constraint/single_stage/constraint_decompose \
  --input test/Passes/mm_2Dmesh/IR/12_after_factorize.mlir \
  > test/Passes/mm_2Dmesh/IR/13_after_decompose.mlir; then
    echo "Error: Step 8 failed."
    exit 1
fi

echo "9) Linearize constraints (McCormick relaxation)..."
if ! ./build/tool/loom-constraint/single_stage/constraint_linearize \
  --input test/Passes/mm_2Dmesh/IR/13_after_decompose.mlir \
  > test/Passes/mm_2Dmesh/IR/14_after_linearize.mlir; then
    echo "Error: Step 9 failed."
    exit 1
fi

echo "10) Compress intermediate variables..."
if ! ./build/tool/loom-constraint/single_stage/compress_intermediate_var \
  --input test/Passes/mm_2Dmesh/IR/14_after_linearize.mlir \
  > test/Passes/mm_2Dmesh/IR/15_after_compress_iv.mlir; then
    echo "Error: Step 10 failed."
    exit 1
fi

echo "11) Simplify constraint space..."
if ! ./build/tool/loom-constraint/single_stage/constraint_simplify \
  --input test/Passes/mm_2Dmesh/IR/15_after_compress_iv.mlir \
  > test/Passes/mm_2Dmesh/IR/16_after_simplify.mlir \
  2> test/Passes/mm_2Dmesh/constraint_space/linearized_constraint_space.json; then
    echo "Error: Step 11 failed."
    exit 1
fi

echo "Lowering Succeed"

echo "1) Visualize Raw Constraint Space..."
if ! python -m lib.lcs.viz_engine.cli \
  --func-name matmul_kernel__d0i0_d1i1__f01__d_d \
  test/Passes/mm_2Dmesh/constraint_space/raw_constraint_space.json \
  --resolution 40 \
  --output test/Passes/mm_2Dmesh/viz/raw_constraint_space.html; then
    echo "Error: Raw visualization failed."
    exit 1
fi

echo "2) Visualize Linear Constraint Space..."
if ! python -m lib.lcs.viz_engine.cli \
  --func-name matmul_kernel__d0i0_d1i1__f01__d_d \
  test/Passes/mm_2Dmesh/constraint_space/linearized_constraint_space.json \
  --resolution 40 \
  --output test/Passes/mm_2Dmesh/viz/linearized_constraint_space.html; then
    echo "Error: Linear visualization failed."
    exit 1
fi

echo "All steps completed successfully."
