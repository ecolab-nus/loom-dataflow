#map = affine_map<()[s0] -> ((s0 + 31) floordiv 32)>
#map1 = affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9, %arg10) = (0, 0) to (%arg6, %arg7) {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
      linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
      %0 = affine.apply #map()[%arg5]
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
      memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %1 = scf.for %arg11 = %c0 to %0 step %c1 iter_args(%arg12 = %alloc_0) -> (memref<32x32xf32>) {
        %3 = affine.apply #map1(%arg11, %arg9)
        %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
        %alloc_2 = memref.alloc() : memref<32x32xf32>
        memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
        %4 = affine.apply #map1(%arg10, %arg11)
        %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%4], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
        %alloc_4 = memref.alloc() : memref<32x32xf32>
        memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
        %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
        memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
        linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
        linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg12, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg12 : memref<32x32xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %5 = arith.addf %in, %in_6 : f32
          linalg.yield %5 : f32
        }
        scf.yield %arg12 : memref<32x32xf32>
      }
      %2 = affine.apply #map1(%arg10, %arg9)
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%2], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
      memref.copy %1, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
    }
    return
  }
}

