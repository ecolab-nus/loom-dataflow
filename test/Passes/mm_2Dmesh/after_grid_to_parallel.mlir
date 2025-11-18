module {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6, %arg7) = (0, 0) to (%arg3, %arg4) {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = tensor.empty() : tensor<64x64xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      affine.for %arg8 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
        affine.for %arg9 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
          %2 = affine.apply affine_map<(d0, d1) -> (d0 * 64 + d1)>(%arg8, %arg6)
          %3 = affine.apply affine_map<(d0, d1) -> (d0 * 64 + d1)>(%arg9, %arg7)
          %4 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %1) -> (tensor<64x64xf32>) {
            %6 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %2)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%6], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            %alloc = memref.alloc() : memref<64x64xf32>
            memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
            %7 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
            %8 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%3, %arg10)
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            %alloc_2 = memref.alloc() : memref<64x64xf32>
            memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
            %9 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
            %10 = linalg.matmul ins(%7, %9 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %10 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
            ^bb0(%in: f32, %in_3: f32, %out: f32):
              %12 = arith.addf %in, %in_3 : f32
              linalg.yield %12 : f32
            } -> tensor<64x64xf32>
            scf.yield %11 : tensor<64x64xf32>
          }
          %5 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%3, %2)
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%5], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          bufferization.materialize_in_destination %4 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
        }
      }
    }
    return
  }
}
