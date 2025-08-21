module {
  // Analysis-only clone of a Triton kernel.
  // Constraints:
  //   - Outer loop: affine.parallel over cores (r, c)
  //   - Body control: affine.for / affine.if only
  //   - Non-control ops: ONLY affine.load / affine.store
  // Symbols (index-typed):
  //   R, C      : grid dims (#rows, #cols of cores)
  //   N, M      : problem dims (rows, cols)
  //   Ti, Tj    : tile sizes per core (use constants below for simplicity)
  func.func @analysis_kernel(
      %A: memref<?x?xf32>, %B: memref<?x?xf32>,
      %R: index, %C: index, %N: index, %M: index) {

    // Each (r, c) is an independent core.
    affine.parallel (%r, %c) = (0, 0) to (%R, %C) {
      // Local tile loops inside a core.
      affine.for %i = 0 to 32 {
        affine.for %j = 0 to 32 {
          // row = r*32 + i, col = c*32 + j
          %v = affine.load %A[%r * 32 + %i, %c * 32 + %j] : memref<?x?xf32>
          affine.store %v, %B[%r * 32 + %i, %c * 32 + %j] : memref<?x?xf32>
        }
      }
    }

    return
  }
}
