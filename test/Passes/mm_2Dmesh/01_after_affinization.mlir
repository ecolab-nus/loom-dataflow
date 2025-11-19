module {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %2 = scf.for %arg9 = %c0 to %c8 step %c1 iter_args(%arg10 = %1) -> (tensor<64x64xf32>) {
      %4 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg6)
      %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
      %alloc = memref.alloc() : memref<64x64xf32>
      memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
      %5 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
      %6 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg7, %arg9)
      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%6], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
      %alloc_2 = memref.alloc() : memref<64x64xf32>
      memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
      %7 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
      %8 = linalg.matmul ins(%5, %7 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %8 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg10 : tensor<64x64xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %10 = arith.addf %in, %in_3 : f32
        linalg.yield %10 : f32
      } -> tensor<64x64xf32>
      scf.yield %9 : tensor<64x64xf32>
    }
    %3 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg7, %arg6)
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%3], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
    return
  }
}
