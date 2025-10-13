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

# Single-stage alloc/copy mapping explorer

This directory contains example inputs to try the `explore_alloc_copy_mapping` single-stage tool.

Usage:

```bash
# Analysis-only: just attach tmd.copy.candidates (no clones)
$ build/tool/triton-shared/single_stage/explore_alloc_copy_mapping \
  --input test/Passes/mm_2Dmesh/reuse_annotated.mlir --analysis-only

# Enumeration: produce function clones per combination (cap to 8 variants)
$ build/tool/triton-shared/single_stage/explore_alloc_copy_mapping \
  --input test/Passes/mm_2Dmesh/reuse_annotated.mlir --max-variants=8
```

Tips:
- The input must already include DF declarations (`df.spatial_dim`, `df.memory`, `df.interconnects`).
- The tool internally runs the reuse annotation pass before exploring mapping.
