# Triton-shared exploration example

Reproduce the exploration pipeline using the sample inputs in this directory.

## How to run
```bash
# From the repository root (after building with ./build.sh)
build/tool/triton_shared_explore \
  --ttshared test/Dialect/Triton/mm_normal/ttshared.mlir \
  --df test/Dialect/DataflowDialect/2D_mesh.mlir \
  > output.mlir
```

## Expected result
The tool should emit the same IR as `expected_output.mlir`. You can diff the files to confirm:
```bash
diff -u test/Passes/explore_over_triton_shared/expected_output.mlir output.mlir
```
