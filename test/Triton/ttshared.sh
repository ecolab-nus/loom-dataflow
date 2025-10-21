export LLVM_BINARY_DIR=<path-to-your-llvm-binaries>
export TRITON_SHARED_OPT_PATH=$TRITON_PLUGIN_DIRS/triton/build/<your-cmake-directory>/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
export TRITON_SHARED_DUMP_PATH=/tmp/triton_ir
export TRITON_PLUGIN_DIRES=<path-to-triton-shared>

/opt/llvm-mlir/bin/mlir-opt \
  -one-shot-bufferize="bufferize-function-boundaries" \
  tmd/test/Dialect/Triton/mm_fixed_strides/ttshared.mlir \
  -o tmd/test/Dialect/Triton/mm_fixed_strides/ttshared.bufferized.mlir


rm -rf ~/.triton/cache/*
# in case of using triton venv
source <path-to-triton-venv>/.venv/bin/activate
python3 <path-to-mm.py>

# Run ttshared-opt
build/tool/ttshared-opt \
  --ttshared test/Dialect/Triton/mm_normal/ttshared.mlir \
  --df test/Dialect/DataflowDialect/2D_mesh.mlir > merged.mlir