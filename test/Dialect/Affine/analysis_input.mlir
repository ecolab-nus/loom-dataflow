module {
  // Number of 128-sized blocks: floor(d/128)
  #nblk = affine_map<(d0) -> (d0 floordiv 128)>
  // Convert a block index to an element offset: i * 128
  #off128 = affine_map<(d0) -> (d0 * 128)>

  func.func @vec_mm(
      %A: memref<?x?xf32>,     // M x K
      %B: memref<?x?xf32>,     // K x N
      %C: memref<?x?xf32>,     // M x N
      %M: index, %N: index, %K: index) {

    // Compute block counts along each dimension.
    %MB = affine.apply #nblk(%M)
    %NB = affine.apply #nblk(%N)
    %KB = affine.apply #nblk(%K)

    // Outer block grid over M and N (in units of 128x128 tiles).
    affine.parallel (%bm, %bn) = (0, 0) to (%MB, %NB) step (1, 1) {
      %m0 = affine.apply #off128(%bm)
      %n0 = affine.apply #off128(%bn)

      // 128x128 view of C (strided layout because base is dynamic).
      %Ctile = memref.subview %C[%m0, %n0] [128, 128] [1, 1]
               : memref<?x?xf32>
                 to memref<128x128xf32, strided<[?, ?], offset: ?>>

      // Sweep K in 128-sized blocks.
      affine.for %bk = 0 to %KB {
        %k0 = affine.apply #off128(%bk)

        // A tile: 128 x 128 (rows from M, cols from K)
        %Atile = memref.subview %A[%m0, %k0] [128, 128] [1, 1]
                 : memref<?x?xf32>
                   to memref<128x128xf32, strided<[?, ?], offset: ?>>

        // B tile: 128 x 128 (rows from K, cols from N)
        %Btile = memref.subview %B[%k0, %n0] [128, 128] [1, 1]
                 : memref<?x?xf32>
                   to memref<128x128xf32, strided<[?, ?], offset: ?>>

        // Micro-kernel: Ctile := Atile * Btile + Ctile
        linalg.matmul
          ins(%Atile, %Btile
              : memref<128x128xf32, strided<[?, ?], offset: ?>>,
                memref<128x128xf32, strided<[?, ?], offset: ?>>)
          outs(%Ctile
              : memref<128x128xf32, strided<[?, ?], offset: ?>>)
      }
    }
    return
  }
}
