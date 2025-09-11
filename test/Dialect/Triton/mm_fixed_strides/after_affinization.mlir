#map = affine_map<()[s0] -> ((s0 + 31) floordiv 32)>
#map1 = affine_map<(d0, d1) -> (d1 * 32768 + d0 * 32)>
#map2 = affine_map<(d0, d1) -> (d1 * 16384 + d0 * 64)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>
module {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = affine.apply #map()[%arg5]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %3 = scf.for %arg12 = %c0 to %2 step %c1 iter_args(%arg13 = %1) -> (tensor<64x64xf32>) {
      %5 = affine.apply #map1(%arg12, %arg9)
      %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
      %alloc = memref.alloc() : memref<64x32xf32>
      memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
      %6 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
      %7 = affine.apply #map2(%arg10, %arg12)
      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%7], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
      %alloc_2 = memref.alloc() : memref<32x64xf32>
      memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
      %8 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
      %9 = linalg.matmul ins(%6, %8 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %10 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg13, %9 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg13 : tensor<64x64xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %11 = arith.addf %in, %in_3 : f32
        linalg.yield %11 : f32
      } -> tensor<64x64xf32>
      scf.yield %10 : tensor<64x64xf32>
    }
    %4 = affine.apply #map4(%arg10, %arg9)
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%4], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
    return
  }
}

