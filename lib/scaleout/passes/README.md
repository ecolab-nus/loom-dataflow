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

## Data reuse analysis
The data reuse analysis for array access.
Considering the array access function of one affine.load `f(i) = Ai + a`. Where `i` is the vector of the iterators, `A` is the vector of coefficients for each iterator, and `a` is a vector of constant. I want to find the invariant (reuse) of the same element of the array, i.e. a `s` such as `f(i+s) = f(i)`. 
Moreover, `f(i+s) = f(i) <=> As = 0`, so actually i want to find all the `s` such as `As = 0`. This is also known ans the integer nullspace or the integer kernel.
 There is an infinity of `s`, but i only need the smallest one, or also known as the primitive vector, i.e. `gcd(s1, s2, ..., sd) = 1`.