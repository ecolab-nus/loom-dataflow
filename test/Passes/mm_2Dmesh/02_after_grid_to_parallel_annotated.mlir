module {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) {
    %c1 = arith.constant 1 : index  // grid_x (placeholder)
    %c1_0 = arith.constant 1 : index  // grid_y (placeholder)
    affine.parallel (%arg8, %arg9) = (0, 0) to (%c1, %c1_0) {  // spatial: core indices
      %cst = arith.constant 0.000000e+00 : f32
      %0 = tensor.empty() : tensor<64x64xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1_1 = arith.constant 1 : index
      %c1_2 = arith.constant 1 : index  // grid_x_inner
      %c1_3 = arith.constant 1 : index  // grid_y_inner
      %c64 = arith.constant 64 : index  // block_m_size
      %c64_4 = arith.constant 64 : index  // block_n_size
      
      // Calculate size per core
      %core_m_size = affine.apply affine_map<(d0, d1) -> (d0 ceildiv d1)>(%arg3, %c1_2)  // M / grid_x
      %core_n_size = affine.apply affine_map<(d0, d1) -> (d0 ceildiv d1)>(%arg4, %c1_3)  // N / grid_y
      
      // Calculate temporal iterations
      %temporal_iter_m = affine.apply affine_map<(d0, d1) -> (d0 ceildiv d1)>(%core_m_size, %c64)  // (M/grid_x) / 64
      %temporal_iter_n = affine.apply affine_map<(d0, d1) -> (d0 ceildiv d1)>(%core_n_size, %c64_4)  // (N/grid_y) / 64
      
      // Temporal loops
      affine.for %arg10 = 0 to affine_map<(d0) -> (d0)>(%temporal_iter_m) {
        affine.for %arg11 = 0 to affine_map<(d0) -> (d0)>(%temporal_iter_n) {
          // Calculate global block indices
          %core_block_m = affine.apply affine_map<(d0, d1) -> (d0 * d1)>(%arg8, %temporal_iter_m)  // core_idx * blocks_per_core
          %global_block_m = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%core_block_m, %arg10)  // + temporal_idx
          %element_idx_m = affine.apply affine_map<(d0, d1) -> (d0 * d1)>(%global_block_m, %c64)  // * block_size
          
          %core_block_n = affine.apply affine_map<(d0, d1) -> (d0 * d1)>(%arg9, %temporal_iter_n)
          %global_block_n = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%core_block_n, %arg11)
          %element_idx_n = affine.apply affine_map<(d0, d1) -> (d0 * d1)>(%global_block_n, %c64_4)
          
          %12 = scf.for %arg12 = %c0 to %c8 step %c1_1 iter_args(%arg13 = %1) -> (tensor<64x64xf32>) {
            %14 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg12, %element_idx_m)
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%14], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            %alloc = memref.alloc() : memref<64x64xf32>
            memref.copy %reinterpret_cast_5, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
            %15 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
            %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%element_idx_n, %arg12)
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            %alloc_7 = memref.alloc() : memref<64x64xf32>
            memref.copy %reinterpret_cast_6, %alloc_7 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
            %17 = bufferization.to_tensor %alloc_7 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
            %18 = linalg.matmul ins(%15, %17 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %18 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg13 : tensor<64x64xf32>) {
            ^bb0(%in: f32, %in_8: f32, %out: f32):
              %20 = arith.addf %in, %in_8 : f32
              linalg.yield %20 : f32
            } -> tensor<64x64xf32>
            scf.yield %19 : tensor<64x64xf32>
          }
          %13 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%element_idx_n, %element_idx_m)
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          bufferization.materialize_in_destination %12 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
        }
      }
    }
    return
  }
}

