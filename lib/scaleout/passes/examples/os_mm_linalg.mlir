module {
  func.func @mm_analysis_one_output_per_core(
      %A: memref<?x?xf32>,     // M x K
      %B: memref<?x?xf32>,     // K x N
      %C: memref<?x?xf32>,     // M x N
      %M: index, %N: index, %K: index) {

    // Hardware-agnostic: this "grid" is the output index space itself.
    // (Mapping to real hardware happens later in your compiler.)
    affine.parallel (%m, %n) = (0, 0) to (%M, %N) {
      affine.for %k = 0 to %K {
        %a = affine.load %A[%m, %k] : memref<?x?xf32>
        %b = affine.load %B[%k, %n] : memref<?x?xf32>
        %c = affine.load %C[%m, %n] : memref<?x?xf32>
        affine.store %c, %C[%m, %n] : memref<?x?xf32>
      }
    }
    return
  }
}