module {
  %0 = df.mat "FPU" {shape = [32, 32, 32]}
  %1 = df.vec "SFPU" {shape = [32]}
  %2 = df.spatial_dim "x", 8
  %3 = df.spatial_dim "y", 8
  %4 = df.core "core" {scaleout=(%2, %3) , scalein=(%0, %1, [8, 1])}
  %5 = df.memory "L1" {scaleout=(%2, %3) , size = 32768, bandwidth = 64}
  %6 = df.mux %4 : !df.compute, %5 : !df.memory  {map = affine_map<(d0, d1) -> (d0, d1)>}
  %7 = df.interconnects "horizontal_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>, spatial_dims = [@y]} : !df.interconnect
  %8 = df.interconnects "vertical_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 8)>, spatial_dims = [@x]} : !df.interconnect
  %9 = df.spatial_dim "d", 4
  %10 = df.memory "DRAM" {scaleout=(%9) , size = 34359738368, bandwidth = 512}
  %11 = df.interconnects "NoC" %5 : !df.memory, %10 : !df.memory  {map = affine_map<(d0, d1) -> (d0 ceildiv 4 + (d1 ceildiv 4) * 2)>} : !df.interconnect
  func.func @matmul_kernel__d0i0_d1i0__f01__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3] {
          affine.for %arg12 = 0 to %arg4 {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg11, %arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg12, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2 * 8 + d1) * 32768 + d0 * 64)>(%arg12, %arg11, %arg10, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f01__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3] {
          affine.for %arg12 = 0 to %arg4 {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg11, %arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg12, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2 * 8 + d1) * 32768 + d0 * 64)>(%arg12, %arg11, %arg10, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f01__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3] {
          affine.for %arg12 = 0 to %arg4 {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg11, %arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg12, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2 * 8 + d1) * 32768 + d0 * 64)>(%arg12, %arg11, %arg10, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to %arg4 {
          affine.for %arg12 = 0 to affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg12, %arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg11, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2 * 8 + d1) * 32768 + d0 * 64)>(%arg11, %arg12, %arg10, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to %arg4 {
          affine.for %arg12 = 0 to affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg12, %arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg11, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2 * 8 + d1) * 32768 + d0 * 64)>(%arg11, %arg12, %arg10, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to %arg4 {
          affine.for %arg12 = 0 to affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg12, %arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg11, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2 * 8 + d1) * 32768 + d0 * 64)>(%arg11, %arg12, %arg10, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg11, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg11, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg11, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg11, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg11, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg11, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__h_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg11, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg11, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg12, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg12, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg12, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg12, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg12, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg12, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__h_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg12, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg12, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg11, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg11, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg11, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg11, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg11, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg11, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__v_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg11, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg11, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg12, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg12, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg12, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg12, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg12, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg12, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__v_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg3] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> ((d2 * 8 + d1) * 32768 + d0 * 64)>(%arg13, %arg12, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d2 * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d3 * 8 + d2) * 32768 + (d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg12, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to %arg3 {
          affine.for %arg12 = 0 to affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg4] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg13, %arg11)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d3 * 32768 + (d2 * 8 + d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg9, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d3 * 32768 + (d2 * 8 + d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to %arg3 {
          affine.for %arg12 = 0 to affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg4] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg13, %arg11)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d3 * 32768 + (d2 * 8 + d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg9, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d3 * 32768 + (d2 * 8 + d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to %arg3 {
          affine.for %arg12 = 0 to affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg4] {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg13, %arg11)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d3 * 32768 + (d2 * 8 + d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg9, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d3 * 32768 + (d2 * 8 + d1 * 8 + d0) * 64)>(%arg12, %arg10, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg4] {
          affine.for %arg12 = 0 to %arg3 {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg13, %arg12)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d3 * 32768 + (d2 * 8 + d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg9, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d3 * 32768 + (d2 * 8 + d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg9, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg4] {
          affine.for %arg12 = 0 to %arg3 {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg13, %arg12)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d3 * 32768 + (d2 * 8 + d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg9, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d3 * 32768 + (d2 * 8 + d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg9, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg4] {
          affine.for %arg12 = 0 to %arg3 {
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %14 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg13, %arg12)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d3 * 32768 + (d2 * 8 + d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg9, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d3 * 32768 + (d2 * 8 + d1 * 8 + d0) * 64)>(%arg11, %arg10, %arg9, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
}
