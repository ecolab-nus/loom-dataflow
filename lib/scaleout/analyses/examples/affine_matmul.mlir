module {
  func.func @matmul_affine(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cf0 = arith.constant 0.0 : f32

    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 {
        affine.store %cf0, %C[%i, %j] : memref<4x4xf32>
        affine.for %k = 0 to 4 {
          %a = affine.load %A[%i, %k] : memref<4x4xf32>
          %b = affine.load %B[%k, %j] : memref<4x4xf32>
          %c = affine.load %C[%i, %j] : memref<4x4xf32>
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %c, %p : f32
          affine.store %s, %C[%i, %j] : memref<4x4xf32>
        }
      }
    }

    return
  }
}


