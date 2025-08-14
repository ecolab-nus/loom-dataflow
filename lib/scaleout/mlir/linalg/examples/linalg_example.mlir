module {
  func.func @matmul_generic(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    linalg.generic
      { indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                         affine_map<(i, j, k) -> (k, j)>,
                         affine_map<(i, j, k) -> (i, j)>],
        iterator_types = ["parallel", "parallel", "reduction"] }
      ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
      outs(%C : memref<4x4xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %prod = arith.mulf %a, %b : f32
        %sum  = arith.addf %c, %prod : f32
        linalg.yield %sum : f32
    }
    return
  }
}


