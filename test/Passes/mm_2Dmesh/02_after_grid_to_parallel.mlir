module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = df.compute "cores", %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %3 = df.memory "L1", %0, %1 {bandwidth = 64 : i64, map = affine_map<(d0, d1) -> (d0, d1)>, size = 32768 : i64}
  %4 = df.mux %2 : !df.compute, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %5 = df.interconnects "horizontal_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !df.interconnect
  %6 = df.interconnects "vertical_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : !df.interconnect
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) {
    %c64 = arith.constant 64 : index
    affine.parallel (%arg8, %arg9) = (0, 0) to (8, 8) {
      %cst = arith.constant 0.000000e+00 : f32
      %7 = tensor.empty() : tensor<64x64xf32>
      %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %9 = affine.apply affine_map<(d0) -> (d0)>(%arg3)
      %10 = affine.apply affine_map<(d0) -> (d0)>(%arg4)
      %11 = affine.apply affine_map<(d0, d1) -> (d0 ceildiv d1)>(%9, %c64)
      %12 = affine.apply affine_map<(d0, d1) -> (d0 ceildiv d1)>(%10, %c64)
      affine.for %arg10 = 0 to affine_map<(d0) -> (d0)>(%11) {
        affine.for %arg11 = 0 to affine_map<(d0) -> (d0)>(%12) {
          %13 = affine.apply affine_map<(d0, d1) -> (d0 * d1)>(%arg8, %11)
          %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%13, %arg10)
          %15 = affine.apply affine_map<(d0, d1) -> (d0 * d1)>(%14, %c64)
          %16 = affine.apply affine_map<(d0, d1) -> (d0 * d1)>(%arg9, %12)
          %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%16, %arg11)
          %18 = affine.apply affine_map<(d0, d1) -> (d0 * d1)>(%17, %c64)
          %19 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %8) -> (tensor<64x64xf32>) {
            %21 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg12, %15)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            %alloc = memref.alloc() : memref<64x64xf32>
            memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
            %22 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
            %23 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%18, %arg12)
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            %alloc_2 = memref.alloc() : memref<64x64xf32>
            memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
            %24 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
            %25 = linalg.matmul ins(%22, %24 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %25 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg13 : tensor<64x64xf32>) {
            ^bb0(%in: f32, %in_3: f32, %out: f32):
              %27 = arith.addf %in, %in_3 : f32
              linalg.yield %27 : f32
            } -> tensor<64x64xf32>
            scf.yield %26 : tensor<64x64xf32>
          }
          %20 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%18, %15)
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          bufferization.materialize_in_destination %19 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
        }
      }
    }
    return
  }
}
