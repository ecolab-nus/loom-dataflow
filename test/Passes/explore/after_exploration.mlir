module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = df.compute "cluster_cores", %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %3 = df.memory "cluster_mem", %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %4 = df.mux %2 : !df.compute, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %5 = df.interconnects %2 : !df.compute, %2 : !df.compute, %0, %1 {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !df.interconnect
  %6 = df.interconnects %2 : !df.compute, %2 : !df.compute, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : !df.interconnect
  func.func @matmul_kernel__d0i0_d1i0_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> (d1)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %9 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %8) -> (tensor<32x32xf32>) {
              %12 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg13, %arg12, %arg9, %arg11)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %13 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %14 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>(%arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %16 = linalg.matmul ins(%13, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg14 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %18 = arith.addf %in, %in_3 : f32
                linalg.yield %18 : f32
              } -> tensor<32x32xf32>
              scf.yield %17 : tensor<32x32xf32>
            }
            %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg10, %arg12, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> (d1)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %9 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %8) -> (tensor<32x32xf32>) {
              %12 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg13, %arg12, %arg10, %arg11)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %13 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %14 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>(%arg9, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %16 = linalg.matmul ins(%13, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg14 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %18 = arith.addf %in, %in_3 : f32
                linalg.yield %18 : f32
              } -> tensor<32x32xf32>
              scf.yield %17 : tensor<32x32xf32>
            }
            %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg9, %arg12, %arg10, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> (d1)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %9 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %8) -> (tensor<32x32xf32>) {
              %12 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg13, %arg12, %arg9, %arg11)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %13 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %14 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>(%arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %16 = linalg.matmul ins(%13, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg14 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %18 = arith.addf %in, %in_3 : f32
                linalg.yield %18 : f32
              } -> tensor<32x32xf32>
              scf.yield %17 : tensor<32x32xf32>
            }
            %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg10, %arg12, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> (d1)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %9 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %8) -> (tensor<32x32xf32>) {
              %12 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg13, %arg12, %arg10, %arg11)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %13 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %14 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>(%arg9, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %16 = linalg.matmul ins(%13, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg14 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %18 = arith.addf %in, %in_3 : f32
                linalg.yield %18 : f32
              } -> tensor<32x32xf32>
              scf.yield %17 : tensor<32x32xf32>
            }
            %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg9, %arg12, %arg10, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %9 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %8) -> (tensor<32x32xf32>) {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 131072 + d2 * 16384)>(%arg13, %arg9, %arg12)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %13 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 256 + d2 * 32)>(%arg13, %arg10, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %16 = linalg.matmul ins(%13, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg14 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %18 = arith.addf %in, %in_3 : f32
                linalg.yield %18 : f32
              } -> tensor<32x32xf32>
              scf.yield %17 : tensor<32x32xf32>
            }
            %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg10, %arg11, %arg9, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %9 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %8) -> (tensor<32x32xf32>) {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 131072 + d2 * 16384)>(%arg13, %arg10, %arg12)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %13 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 256 + d2 * 32)>(%arg13, %arg9, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %16 = linalg.matmul ins(%13, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg14 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %18 = arith.addf %in, %in_3 : f32
                linalg.yield %18 : f32
              } -> tensor<32x32xf32>
              scf.yield %17 : tensor<32x32xf32>
            }
            %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg9, %arg11, %arg10, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %9 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %8) -> (tensor<32x32xf32>) {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 131072 + d2 * 16384)>(%arg13, %arg9, %arg12)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %13 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 256 + d2 * 32)>(%arg13, %arg10, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %16 = linalg.matmul ins(%13, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg14 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %18 = arith.addf %in, %in_3 : f32
                linalg.yield %18 : f32
              } -> tensor<32x32xf32>
              scf.yield %17 : tensor<32x32xf32>
            }
            %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg10, %arg11, %arg9, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %9 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %8) -> (tensor<32x32xf32>) {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 131072 + d2 * 16384)>(%arg13, %arg10, %arg12)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %13 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 256 + d2 * 32)>(%arg13, %arg9, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %16 = linalg.matmul ins(%13, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg14 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %18 = arith.addf %in, %in_3 : f32
                linalg.yield %18 : f32
              } -> tensor<32x32xf32>
              scf.yield %17 : tensor<32x32xf32>
            }
            %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg9, %arg11, %arg10, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> (d0)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %9 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %8) -> (tensor<32x32xf32>) {
              %12 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>(%arg13, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %13 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %14 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg13, %arg12, %arg10, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %16 = linalg.matmul ins(%13, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg14 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %18 = arith.addf %in, %in_3 : f32
                linalg.yield %18 : f32
              } -> tensor<32x32xf32>
              scf.yield %17 : tensor<32x32xf32>
            }
            %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg9, %arg12, %arg10, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> (d0)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %9 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %8) -> (tensor<32x32xf32>) {
              %12 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>(%arg13, %arg10)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %13 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %14 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg13, %arg12, %arg9, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %16 = linalg.matmul ins(%13, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg14 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %18 = arith.addf %in, %in_3 : f32
                linalg.yield %18 : f32
              } -> tensor<32x32xf32>
              scf.yield %17 : tensor<32x32xf32>
            }
            %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg10, %arg12, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> (d0)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %9 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %8) -> (tensor<32x32xf32>) {
              %12 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>(%arg13, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %13 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %14 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg13, %arg12, %arg10, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %16 = linalg.matmul ins(%13, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg14 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %18 = arith.addf %in, %in_3 : f32
                linalg.yield %18 : f32
              } -> tensor<32x32xf32>
              scf.yield %17 : tensor<32x32xf32>
            }
            %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg9, %arg12, %arg10, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> (d0)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %7 = tensor.empty() : tensor<32x32xf32>
            %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %9 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %10 = scf.for %arg13 = %c0 to %9 step %c1 iter_args(%arg14 = %8) -> (tensor<32x32xf32>) {
              %12 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>(%arg13, %arg10)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %13 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %14 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg13, %arg12, %arg9, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
              %16 = linalg.matmul ins(%13, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg14 : tensor<32x32xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %18 = arith.addf %in, %in_3 : f32
                linalg.yield %18 : f32
              } -> tensor<32x32xf32>
              scf.yield %17 : tensor<32x32xf32>
            }
            %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg10, %arg12, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %10 in writable %reinterpret_cast : (tensor<32x32xf32>, memref<32x32xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
}
