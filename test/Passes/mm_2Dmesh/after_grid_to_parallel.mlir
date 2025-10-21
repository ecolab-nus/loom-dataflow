module {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.parallel (%arg12, %arg13, %arg14) = (0, 0, 0) to (%arg9, %arg10, %arg11) {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = tensor.empty() : tensor<64x64xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %c64 = arith.constant 64 : index
      %2 = arith.muli %arg12, %c64 : index
      %3 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %4 = scf.for %arg15 = %c0 to %3 step %c1 iter_args(%arg16 = %1) -> (tensor<64x64xf32>) {
        %c32 = arith.constant 32 : index
        %11 = arith.muli %arg15, %c32 : index
        %12 = arith.muli %2, %arg6 : index
        %13 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg15)[%12]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%13], sizes: [64, 32], strides: [%arg6, 1] : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
        %alloc = memref.alloc() : memref<64x32xf32>
        memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
        %14 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
        %15 = arith.muli %11, %arg7 : index
        %16 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%15]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%16], sizes: [32, 64], strides: [%arg7, 1] : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
        %alloc_2 = memref.alloc() : memref<32x64xf32>
        memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
        %17 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
        %18 = linalg.matmul ins(%14, %17 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %18 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg16 : tensor<64x64xf32>) {
        ^bb0(%in: f32, %in_3: f32, %out: f32):
          %20 = arith.addf %in, %in_3 : f32
          linalg.yield %20 : f32
        } -> tensor<64x64xf32>
        scf.yield %19 : tensor<64x64xf32>
      }
      %5 = arith.muli %2, %arg8 : index
      %6 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%5]
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [64, 64], strides: [%arg8, 1] : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %7 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %arg13, %arg14)[%arg3]
      %8 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %arg13, %arg14)[%arg4]
      %extracted_slice = tensor.extract_slice %4[0, 0] [%7, %8] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
      %9 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %arg13, %arg14)[%arg3]
      %10 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %arg13, %arg14)[%arg4]
      %subview = memref.subview %reinterpret_cast[0, 0] [%9, %10] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }
}
