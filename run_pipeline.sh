#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

EXAMPLE=$1
RUN_STEPS=$2

if [ -z "$EXAMPLE" ]; then
    echo "Usage: $0 <example> \"[1,2,3,4,5,6,7,8,9]\""
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

echo "Creating necessary directories for $EXAMPLE..."
mkdir -p examples/$EXAMPLE/IR
mkdir -p examples/$EXAMPLE/constraint_space

if should_run 1; then
    echo "1) Specialize linalg operations' destination..."
    if ! build/tool/loom-opt/single_stage/tensor_canonicalize \
      --input examples/$EXAMPLE/IR/00_from_helion_frontend.mlir  \
      > examples/$EXAMPLE/IR/01_tensor_canonicalized.mlir; then
        echo "Error: Step 1 failed."
        exit 1
    fi
fi

if should_run 2; then
    echo "2) Replace grid indices with a 3-D affine.parallel..."
    if ! build/tool/loom-opt/single_stage/memory_binding \
      --input examples/$EXAMPLE/IR/01_tensor_canonicalized.mlir  \
      > examples/$EXAMPLE/IR/02_explicit_memory_access.mlir; then
        echo "Error: Step 2 failed."
        exit 1
    fi
fi

if should_run 3; then
    echo "3) Enumerate spatial mappings and merge DF declarations..."
    if ! build/tool/loom-opt/single_stage/enumerate_hw_mapping \
      --input examples/$EXAMPLE/IR/02_explicit_memory_access.mlir \
      --hw_spec ../loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir \
      > examples/$EXAMPLE/IR/03_after_hardware_mapping.mlir; then
        echo "Error: Step 3 failed."
        exit 1
    fi
fi

if should_run 4; then
    echo "4) Analyze reuse pattern on loom.subview..."
    if ! build/tool/loom-opt/single_stage/analyze_reuse \
      --input examples/$EXAMPLE/IR/03_after_hardware_mapping.mlir \
      > examples/$EXAMPLE/IR/04_after_reuse_analyzation.mlir; then
        echo "Error: Step 4 failed."
        exit 1
    fi
fi

if should_run 5; then
    echo "5) Enumerate copy interconnect broadcast choices on loom.copy_to_tensor..."
    if ! build/tool/loom-opt/single_stage/enumerate_copy_broadcast \
      --input examples/$EXAMPLE/IR/04_after_reuse_analyzation.mlir \
      > examples/$EXAMPLE/IR/05_after_enumerate_broadcast.mlir; then
        echo "Error: Step 5 failed."
        exit 1
    fi
fi

if should_run 6; then
    echo "6) Dump ETG..."
    if ! build/tool/loom-opt/single_stage/staged_etg \
      --input examples/$EXAMPLE/IR/05_after_enumerate_broadcast.mlir \
      --hw_spec ../loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir \
      --output examples/$EXAMPLE/constraint_space/staged_etg_dump.json; then
        echo "Error: Step 6 failed."
        exit 1
    fi
fi

if should_run 7; then
    echo "7) Materialize symbolic block sizes..."
    if ! build/tool/loom-opt/single_stage/canonicalize \
      --input examples/$EXAMPLE/IR/05_after_enumerate_broadcast.mlir \
      > examples/$EXAMPLE/IR/06_after_canonicalize.mlir; then
        echo "Error: Step 7 failed."
        exit 1
    fi
fi

if should_run 8; then
    echo "8) OSB..."
    if ! build/tool/loom-opt/single_stage/one_shot_bufferize \
      --input examples/$EXAMPLE/IR/06_after_canonicalize.mlir \
      > examples/$EXAMPLE/IR/07_after_osb.mlir; then
        echo "Error: Step 8 failed."
        exit 1
    fi
fi

if should_run 9; then
    echo "9) tt-opt..."
    if ! build/tool/tt-opt/single_stage/tt-opt \
      --input examples/$EXAMPLE/IR/07_after_osb.mlir \
      > examples/$EXAMPLE/IR/08_tt-opt.mlir; then
        echo "Error: Step 9 failed."
        exit 1
    fi
fi

echo "Pipeline execution completed."
