# Analysis IR Contract for Triton→Dataflow Project
This sub-project defines analysis on affine-based IR, used to detect the data reuse accross different iterations. 
This takes as input nested affine loops where the data-independent loops are affine.parallel, while the data-dependent loops are affine.for.
The loop body only contains memory-related instructions for detecting the data movements. Concretely, you may have:

- **Control flow**
  - `affine.parallel` — outermost grid (e.g., `%r`, `%c` for row/col cores), no data dependency among the iterations
  - `affine.for` — local loops within a core, could have data dependency among the iterations (we encourage to put all data-independent loops in the grid)
  - `affine.if` — structural sparsity of the loop, defines which iterations are jumped
- **Memory**
  - `affine.load`, `affine.store` — **only** non-control ops allowed

**Disallowed** in this IR:
- `scf.*`, `gpu.*`, `affine.apply`, arithmetic/computation ops (`arith.*`, `math.*`), `memref.subview`/casts, etc.

> Rationale: keeping only affine control + affine memory ops lets us form exact Presburger sets/relations for accesses without verifier headaches or partial legality.

## What the analysis computes
1. Per-core footprint for each memref X:
  - Build a relation F_X(r,c,i,j, …; symbols) → indices
  - Project out local IVs (i,j, …) to obtain a set Footprint_X(r,c; symbols)
2. Reuse detection:
  - Row invariance: check whether access functions are independent of c → candidate for row broadcast.
  - Overlap: intersect Footprint_X(r,c) with neighbors (e.g., (r,c+1)) to quantify shared data.
3. Volume/size (simplified tile model):
  - For rectangular tiles, |Footprint_X(r,c)| = min(Ti, N - r*Ti) * min(Tj, M - c*Tj) (with guards).
  - For uniform exact tiling (no boundaries), this simplifies to Ti * Tj.


## Analysis Outputs
The analysis add attributes to the input mlir and makes this attributes printable (in the output mlir file)