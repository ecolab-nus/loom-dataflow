#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

TEST_CASE=$1
RUN_STEPS=$2

if [ -z "$TEST_CASE" ]; then
    echo "Usage: $0 <test_case> \"[1,2,3,4,5,6,7,8,9]\""
    exit 1
fi

# Default to all steps if not provided
if [ -z "$RUN_STEPS" ]; then
    RUN_STEPS="[1,2,3,4,5,6,7,8,9]"
fi

should_run() {
    local step=$1
    if [[ "$RUN_STEPS" =~ "$step" ]]; then
        return 0
    else
        return 1
    fi
}

echo "Starting build..."
if ! ./build.sh; then
    echo "Error: ./build.sh failed."
    exit 1
fi

echo "Creating necessary directories for $TEST_CASE..."
mkdir -p test/Passes/$TEST_CASE/IR
mkdir -p test/Passes/$TEST_CASE/constraint_space

if should_run 1; then
    echo "1) Specialize linalg operations' destination..."
    if ! build/tool/loom-opt/single_stage/tensor_canonicalize \
      --input test/Passes/$TEST_CASE/IR/00_from_helion_frontend.mlir  \
      > test/Passes/$TEST_CASE/IR/01_tensor_canonicalized.mlir; then
        echo "Error: Step 1 failed."
        exit 1
    fi
fi

if should_run 2; then
    echo "2) Replace grid indices with a 3-D affine.parallel..."
    if ! build/tool/loom-opt/single_stage/memory_binding \
      --input test/Passes/$TEST_CASE/IR/01_tensor_canonicalized.mlir  \
      > test/Passes/$TEST_CASE/IR/02_explicit_memory_access.mlir; then
        echo "Error: Step 2 failed."
        exit 1
    fi
fi

if should_run 3; then
    echo "3) Enumerate spatial mappings and merge DF declarations..."
    if ! build/tool/loom-opt/single_stage/enumerate_hw_mapping \
      --input test/Passes/$TEST_CASE/IR/02_explicit_memory_access.mlir \
      --hw_spec ../loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir \
      > test/Passes/$TEST_CASE/IR/03_after_hardware_mapping.mlir; then
        echo "Error: Step 3 failed."
        exit 1
    fi
fi

if should_run 4; then
    echo "4) Analyze reuse pattern on loom.subview..."
    if ! build/tool/loom-opt/single_stage/analyze_reuse \
      --input test/Passes/$TEST_CASE/IR/03_after_hardware_mapping.mlir \
      > test/Passes/$TEST_CASE/IR/04_after_reuse_analyzation.mlir; then
        echo "Error: Step 4 failed."
        exit 1
    fi
fi

if should_run 5; then
    echo "5) Enumerate copy interconnect broadcast choices on loom.copy_to_tensor..."
    if ! build/tool/loom-opt/single_stage/enumerate_copy_broadcast \
      --input test/Passes/$TEST_CASE/IR/04_after_reuse_analyzation.mlir \
      > test/Passes/$TEST_CASE/IR/05_after_enumerate_broadcast.mlir; then
        echo "Error: Step 5 failed."
        exit 1
    fi
fi

if should_run 6; then
    echo "6) Dump ETG..."
    if ! build/tool/loom-opt/single_stage/staged_etg \
      --input test/Passes/$TEST_CASE/IR/05_after_enumerate_broadcast.mlir \
      --hw_spec ../loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir \
      --output test/Passes/$TEST_CASE/constraint_space/staged_etg_dump.json; then
        echo "Error: Step 6 failed."
        exit 1
    fi
fi

if should_run 7; then
    echo "7) Materialize symbolic block sizes..."
    if ! build/tool/loom-opt/single_stage/canonicalize \
      --input test/Passes/$TEST_CASE/IR/05_after_enumerate_broadcast.mlir \
      > test/Passes/$TEST_CASE/IR/06_after_canonicalize.mlir; then
        echo "Error: Step 7 failed."
        exit 1
    fi
fi

if should_run 8; then
    echo "8) OSB..."
    if ! build/tool/loom-opt/single_stage/one_shot_bufferize \
      --input test/Passes/$TEST_CASE/IR/06_after_canonicalize.mlir \
      > test/Passes/$TEST_CASE/IR/07_after_osb.mlir; then
        echo "Error: Step 8 failed."
        exit 1
    fi
fi

if should_run 9; then
    echo "9) tt-opt..."
    if ! build/tool/tt-opt/single_stage/tt-opt \
      --input test/Passes/$TEST_CASE/IR/07_after_osb.mlir \
      > test/Passes/$TEST_CASE/IR/08_tt-opt.mlir; then
        echo "Error: Step 9 failed."
        exit 1
    fi
fi

echo "Pipeline execution completed."
