module attributes {loom.block_k = 64 : index, loom.block_m = 64 : index, loom.block_n = 64 : index} {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) {
    affine.parallel (%arg12, %arg13) = (0, 0) to (%arg3, %arg4) {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = tensor.empty(%arg9, %arg10) : tensor<?x?xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %2 = arith.muli %arg12, %arg9 : index
      %3 = arith.muli %arg13, %arg10 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %4 = scf.for %arg14 = %c0 to %c8 step %c1 iter_args(%arg15 = %1) -> (tensor<?x?xf32>) {
        %6 = arith.muli %arg14, %arg11 : index
        %7 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%6, %2]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [%arg9, %arg11], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %alloc = memref.alloc(%arg9, %arg11) : memref<?x?xf32>
        memref.copy %reinterpret_cast_0, %alloc : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
        %8 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
        %9 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%3, %6]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%9], sizes: [%arg11, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %alloc_2 = memref.alloc(%arg11, %arg10) : memref<?x?xf32>
        memref.copy %reinterpret_cast_1, %alloc_2 : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
        %10 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
        %11 = linalg.matmul ins(%8, %10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %11 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg15 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_3: f32, %out: f32):
          %13 = arith.addf %in, %in_3 : f32
          linalg.yield %13 : f32
        } -> tensor<?x?xf32>
        scf.yield %12 : tensor<?x?xf32>
      }
      %5 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%3, %2]
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%5], sizes: [%arg9, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
      bufferization.materialize_in_destination %4 in writable %reinterpret_cast : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
    }
    return
  }
}
