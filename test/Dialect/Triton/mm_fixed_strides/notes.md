/opt/llvm-mlir/bin/mlir-opt \
  -one-shot-bufferize="bufferize-function-boundaries" \
  tmd/test/Dialect/Triton/mm_fixed_strides/ttshared.mlir \
  -o tmd/test/Dialect/Triton/mm_fixed_strides/ttshared.bufferized.mlir