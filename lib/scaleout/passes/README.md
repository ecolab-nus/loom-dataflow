## Triton Shared Affinize Pass

This pass converts arithmetic index expressions produced by Triton-shared lowered kernels into affine form and replaces eligible memory and loop constructs with their affine counterparts. The goal is to maximize the use of `affine.apply`, `affine.min/max`, `affine.load/store`, and index-typed IR, while eliminating redundant casts and dead arithmetic.

### Summary of transformations

- **Function argument retyping**
  - Convert all `i32` function arguments to `index` to eliminate pervasive `arith.index_cast` in the body and enable direct use in affine expressions.

- **Integer arithmetic harmonization**
  - Rebuild `arith.addi/subi/muli/divsi/minsi/maxsi` where any operand/result is `index` or mixed-typed so that operands/results are of type `index`.
  - Iterate to a fixed point to remove all mixed-type arithmetic introduced by retyping.

- **Index → `affine.apply` conversion**
  - For index-typed SSA values representing affine expressions, inject `affine.apply` and feed users with its result.
  - Partial conversions for `addi/subi` when one side is affine: the affine side is folded into `affine.apply`, the other side remains as-is.

- **`memref.reinterpret_cast` offsets affinization**
  - Rebuild the op with offsets expressed via `affine.apply` when the offset expressions are affine in terms of function args and surrounding loop IVs.
  - Sizes/strides are kept unchanged (focus is offsets) to avoid dominance and aliasing issues.

- **`tensor.extract_slice` / `memref.subview` sizes affinization**
  - Convert dynamic sizes to `affine.apply` where possible.
  - Recognize and replace common clamp patterns with a single `affine.min` over a multi-result affine map:
    - Pattern: `min( max(min(base + T, dim), base) - base, T )` → `affine.min( dim - base, T )`.
  - For `min/max` of affine expressions, build a multi-result affine map and emit `affine.min`/`affine.max` directly.

- **`memref.load/store` → `affine.load/store`**
  - Replace when all indices are either already trivially affine (loop IVs / block args / constant indices) or can be converted to `affine.apply` via the index → affine conversion.

- **`scf.for` normalization (index IV and affine bounds)**
  - Rebuild `scf.for` with index-typed lower bound, upper bound, and step. Clone the loop body so the induction variable is natively type `index` without extra casts.
  - Affinize loop bounds and step with `affine.apply`/`affine.min/max` where possible, so upper bounds can be expressed as affine maps.

- **Redundant cast cleanup and DCE**
  - Remove redundant `arith.index_cast` where source and destination types are the same.
  - General dead code elimination: iteratively erase any trivially-dead operation (using MLIR’s `isOpTriviallyDead`).
  - Iteration guards ensure termination even if follow-on canonicalizations unlock additional cleanup.

### Affine-eligibility and expression model

The pass constructs affine expressions treating:

- **Dims**: function block arguments and surrounding `scf.for` induction variables.
- **Symbols**: other index values that appear in the expression when they cannot be proven as dims (including non-dim block arguments and non-affine sub-expressions promoted to symbols).

An index expression is considered affine if it can be built using the following primitives:

- `addi`, `subi`
- `muli` with a compile-time constant factor
- `divsi` by a non-zero compile-time constant (lowered as floorDiv)
- constants and `arith.index_cast` passthrough

If an expression contains non-affine pieces, the builder promotes that value to a symbol and continues, so it can still participate in an affine map as a symbol operand. For partial conversions of `addi/subi`, if only one side is affine, that side is first expressed via `affine.apply` and recombined with the non-affine side using `arith`.

For `min/max`:

- `affine.min/max` are emitted when both sides are affine (possibly with symbols). Multiple affine results are bundled in a single map and wired into a single `affine.min`/`affine.max`.

### Preconditions for each rewrite

- **Function arg retyping**: any function input of MLIR type `i32` is converted to MLIR type `index`.
- **Arithmetic harmonization**: applies to `arith.addi/subi/muli/divsi/minsi/maxsi` when operand/result types differ or when any of them is `index`.
- **Index → `affine.apply`**: the value must be of type `index` and its expression must satisfy the affine-eligibility rules above. Partial add/sub rewrites require at least one affine side.
- **`reinterpret_cast` offsets**: each offset expression is independently tested for affine-eligibility using dims (grid IDs and surrounding loop IVs) plus promoted symbols.
- **`extract_slice` / `subview` sizes**: each size is independently tested for affine-eligibility; clamp pattern is recognized and rewritten; min/max are emitted if both sides are affine.
- **`memref.load/store` replacement**: all indices must be affine or trivially acceptable (loop IVs / block args / `arith.constant index`).
- **`scf.for`**: loop is rebuilt if any of lb/ub/step/IV is not `index`, after converting them to `index` and affinizing their expressions.

### Limitations and non-goals

- General multiplication of two non-constant terms is not considered affine.
- Non-linear operations other than `min/max` are not converted.
- `memref.reinterpret_cast` sizes/strides are not rewritten (only offsets) to avoid introducing dominance/aliasing issues.
- The pass is conservative; if an expression cannot be proven affine with the allowed primitives, it is left as-is (or partially converted when applicable).

### Dialects and dependencies

The pass requires: `affine`, `arith`, `memref`, `func`, `scf`, `linalg`, `tensor`, `bufferization`.

### Running the pass

- Pass name: `tmd-triton-shared-affinize`
- If integrated into `mlir-opt` or your pipeline runner, use:

```bash
mlir-opt -tmd-triton-shared-affinize input.mlir > output.mlir
```

- With the standalone driver (in this repo):

```bash
build/lib/scaleout/passes/tmd_triton_shared_affinize input.mlir > output.mlir
```

### Examples

#### Loop normalization and affine upper bound

Before:

```mlir
%c0_i32 = arith.constant 0 : i32
%c1_i32 = arith.constant 1 : i32
%t = arith.divsi %x, %c32_i32 : i32
scf.for %iv = %c0_i32 to %t step %c1_i32 { ... }
```

After:

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%2  = affine.apply #map(%x)              // affine upper bound
scf.for %iv = %c0 to %2 step %c1 { ... } // %iv is index
```

#### Clamp pattern in slice sizes

Before:

```mlir
%sz = arith.minsi (arith.subi (arith.maxsi (arith.minsi (arith.addi %base, %T), %dim), %base), %T) : index
%slice = tensor.extract_slice %t[0, 0] [%sz, %sz2] [1, 1] : tensor<...> to tensor<?x?x...>
```

After:

```mlir
%sz = affine.min #map(%dim, %base)[%T]   // map returns (dim - base, T)
%slice = tensor.extract_slice %t[0, 0] [%sz, %sz2] [1, 1] : tensor<...> to tensor<?x?x...>
```

### Notes on robustness

- Iteration guards are used in cleanup loops (cast folding, general DCE, and arithmetic harmonization) to ensure termination even when other canonicalizations unlock more work.
- Symbol ordering is normalized per-map so that the same semantic expression yields stable operand ordering.


