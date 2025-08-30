module {
  func.func @vec_mm(
      %A: memref<?x?xf32>,     // M x K
      %B: memref<?x?xf32>,     // K x N
      %C: memref<?x?xf32>,     // M x N
      %M: index, %N: index, %K: index) {

    affine.parallel (%m_outer, %n_outer) = (0, 0) to (%M, %N) step (128, 128) {
      affine.for %k = 0 to %K {
        affine.parallel (%m_inner, %n_inner) = (0, 0) to (128, 128) step (1, 1) {
          %a = affine.load %A[%m_outer * 128 + %m_inner, %k] : memref<?x?xf32>
          %b = affine.load %B[%k, %n_outer * 128 + %n_inner] : memref<?x?xf32>
          %c = affine.load %C[%m_outer * 128 + %m_inner, %n_outer * 128 + %n_inner] : memref<?x?xf32>
          affine.store %c, %C[%m_outer * 128 + %m_inner, %n_outer * 128 + %n_inner] : memref<?x?xf32>
        }
      }
    }
    return
  }
}