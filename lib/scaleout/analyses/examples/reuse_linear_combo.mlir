module {
  // Test: A[2m + n, k] should have primitive reuse vectors [1, -2, 0] and [-1, 2, 0]
  func.func @reuse_linear_combo(
      %A: memref<?x?xf32>,
      %M: index, %N: index, %K: index) {
    affine.parallel (%m, %n) = (0, 0) to (%M, %N) {
      affine.for %k = 0 to %K {
        %a = affine.load %A[2 * %m + %n, %k] : memref<?x?xf32>
        affine.store %a, %A[2 * %m + %n, %k] : memref<?x?xf32>
      }
    }
    return
  }
}


