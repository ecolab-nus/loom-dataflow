#map = affine_map<(d0) -> (-d0 + 4, 8)>
#map1 = affine_map<(d0) -> (d0 - 1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_generic(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    scf.for %arg3 = %c0 to %c4 step %c8 {
      %c4_0 = arith.constant 4 : index
      %0 = affine.min #map(%arg3)
      %1 = affine.apply #map1(%0)
      %2 = affine.apply #map1(%0)
      %3 = affine.apply #map1(%0)
      %subview = memref.subview %arg0[0, %arg3] [4, %0] [1, 1] : memref<4x4xf32> to memref<4x?xf32, strided<[4, 1], offset: ?>>
      %subview_1 = memref.subview %arg1[%arg3, 0] [%0, 4] [1, 1] : memref<4x4xf32> to memref<?x4xf32, strided<[4, 1], offset: ?>>
      %subview_2 = memref.subview %arg2[0, 0] [4, 4] [1, 1] : memref<4x4xf32> to memref<4x4xf32, strided<[4, 1]>>
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview, %subview_1 : memref<4x?xf32, strided<[4, 1], offset: ?>>, memref<?x4xf32, strided<[4, 1], offset: ?>>) outs(%subview_2 : memref<4x4xf32, strided<[4, 1]>>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %4 = arith.mulf %in, %in_3 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
    }
    return
  }
}
