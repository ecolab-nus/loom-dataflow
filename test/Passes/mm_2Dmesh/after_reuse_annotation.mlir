module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = df.compute "cores", %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %3 = df.memory "L1", %0, %1 {bandwidth = 64 : i64, map = affine_map<(d0, d1) -> (d0, d1)>, size = 8192 : i64}
  %4 = df.mux %2 : !df.compute, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %5 = df.interconnects "horizontal_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !df.interconnect
  %6 = df.interconnects "vertical_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : !df.interconnect
  func.func @matmul_kernel__d0i0_d1i0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %c0 = arith.constant 0 : index
            %c16 = arith.constant 16 : index
            %c1 = arith.constant 1 : index
            %9 = scf.for %arg10 = %c0 to %c16 step %c1 iter_args(%arg11 = %8) -> (tensor<32x32xf32>) {
              %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg10, %arg9, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %12 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %13 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>(%arg7, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %15 = linalg.matmul ins(%12, %14 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %17 = arith.addf %in, %in_3 : f32
                linalg.yield %17 : f32
              } -> tensor<32x32xf32>
              scf.yield %16 : tensor<32x32xf32>
            }
            %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %9 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %c0 = arith.constant 0 : index
            %c16 = arith.constant 16 : index
            %c1 = arith.constant 1 : index
            %9 = scf.for %arg10 = %c0 to %c16 step %c1 iter_args(%arg11 = %8) -> (tensor<32x32xf32>) {
              %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg10, %arg9, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %12 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %13 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>(%arg7, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %15 = linalg.matmul ins(%12, %14 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %17 = arith.addf %in, %in_3 : f32
                linalg.yield %17 : f32
              } -> tensor<32x32xf32>
              scf.yield %16 : tensor<32x32xf32>
            }
            %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %9 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %c0 = arith.constant 0 : index
            %c16 = arith.constant 16 : index
            %c1 = arith.constant 1 : index
            %9 = scf.for %arg10 = %c0 to %c16 step %c1 iter_args(%arg11 = %8) -> (tensor<32x32xf32>) {
              %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 131072 + d2 * 16384)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %12 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %13 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 256 + d2 * 32)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %15 = linalg.matmul ins(%12, %14 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %17 = arith.addf %in, %in_3 : f32
                linalg.yield %17 : f32
              } -> tensor<32x32xf32>
              scf.yield %16 : tensor<32x32xf32>
            }
            %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %9 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %c0 = arith.constant 0 : index
            %c16 = arith.constant 16 : index
            %c1 = arith.constant 1 : index
            %9 = scf.for %arg10 = %c0 to %c16 step %c1 iter_args(%arg11 = %8) -> (tensor<32x32xf32>) {
              %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 131072 + d2 * 16384)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %12 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %13 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 256 + d2 * 32)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %15 = linalg.matmul ins(%12, %14 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %17 = arith.addf %in, %in_3 : f32
                linalg.yield %17 : f32
              } -> tensor<32x32xf32>
              scf.yield %16 : tensor<32x32xf32>
            }
            %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %9 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %c0 = arith.constant 0 : index
            %c16 = arith.constant 16 : index
            %c1 = arith.constant 1 : index
            %9 = scf.for %arg10 = %c0 to %c16 step %c1 iter_args(%arg11 = %8) -> (tensor<32x32xf32>) {
              %11 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>(%arg10, %arg6)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %12 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %13 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg10, %arg9, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %15 = linalg.matmul ins(%12, %14 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %17 = arith.addf %in, %in_3 : f32
                linalg.yield %17 : f32
              } -> tensor<32x32xf32>
              scf.yield %16 : tensor<32x32xf32>
            }
            %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg6, %arg9, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %9 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %c0 = arith.constant 0 : index
            %c16 = arith.constant 16 : index
            %c1 = arith.constant 1 : index
            %9 = scf.for %arg10 = %c0 to %c16 step %c1 iter_args(%arg11 = %8) -> (tensor<32x32xf32>) {
              %11 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>(%arg10, %arg6)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %12 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %13 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg10, %arg9, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %15 = linalg.matmul ins(%12, %14 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %17 = arith.addf %in, %in_3 : f32
                linalg.yield %17 : f32
              } -> tensor<32x32xf32>
              scf.yield %16 : tensor<32x32xf32>
            }
            %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg6, %arg9, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %9 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
}
