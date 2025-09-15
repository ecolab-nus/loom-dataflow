module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = "df.interconnects"(%0, %1) <{map = affine_map<(d0, d1) -> (d0 + 1, d1)>}> : (index, index) -> !df.interconnect
  %3 = "df.interconnects"(%0, %1) <{map = affine_map<(d0, d1) -> (d0, d1 + 1)>}> : (index, index) -> !df.interconnect
  func.func @matmul_kernel__d0i0_d1i0_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg6, %arg7) {
      affine.for %arg10 = 0 to affine_map<(d0, d1) -> (d1)>(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %4 = tensor.empty() : tensor<64x64xf32>
            %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %6 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %7 = scf.for %arg13 = %c0 to %6 step %c1 iter_args(%arg14 = %5) -> (tensor<64x64xf32>) {
              %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32768 + d2 * 2097152 + d3 * 262144)>(%arg13, %arg12, %arg9, %arg11)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
              %10 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
              %11 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 64)>(%arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
              %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
              %13 = linalg.matmul ins(%10, %12 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %15 = arith.addf %in, %in_3 : f32
                linalg.yield %15 : f32
              } -> tensor<64x64xf32>
              scf.yield %14 : tensor<64x64xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 32768 + d2 * 2097152 + d3 * 262144)>(%arg10, %arg12, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
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
            %4 = tensor.empty() : tensor<64x64xf32>
            %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %6 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %7 = scf.for %arg13 = %c0 to %6 step %c1 iter_args(%arg14 = %5) -> (tensor<64x64xf32>) {
              %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32768 + d2 * 2097152 + d3 * 262144)>(%arg13, %arg12, %arg10, %arg11)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
              %10 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
              %11 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 64)>(%arg9, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
              %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
              %13 = linalg.matmul ins(%10, %12 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %15 = arith.addf %in, %in_3 : f32
                linalg.yield %15 : f32
              } -> tensor<64x64xf32>
              scf.yield %14 : tensor<64x64xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 32768 + d2 * 2097152 + d3 * 262144)>(%arg9, %arg12, %arg10, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
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
            %4 = tensor.empty() : tensor<64x64xf32>
            %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %6 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %7 = scf.for %arg13 = %c0 to %6 step %c1 iter_args(%arg14 = %5) -> (tensor<64x64xf32>) {
              %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32768 + d2 * 2097152 + d3 * 262144)>(%arg13, %arg12, %arg9, %arg11)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
              %10 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
              %11 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 64)>(%arg10, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
              %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
              %13 = linalg.matmul ins(%10, %12 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %15 = arith.addf %in, %in_3 : f32
                linalg.yield %15 : f32
              } -> tensor<64x64xf32>
              scf.yield %14 : tensor<64x64xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 32768 + d2 * 2097152 + d3 * 262144)>(%arg10, %arg12, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
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
            %4 = tensor.empty() : tensor<64x64xf32>
            %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %6 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %7 = scf.for %arg13 = %c0 to %6 step %c1 iter_args(%arg14 = %5) -> (tensor<64x64xf32>) {
              %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32768 + d2 * 2097152 + d3 * 262144)>(%arg13, %arg12, %arg10, %arg11)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
              %10 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
              %11 = affine.apply affine_map<(d0, d1) -> (d1 * 16384 + d0 * 64)>(%arg9, %arg13)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
              %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
              %13 = linalg.matmul ins(%10, %12 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %15 = arith.addf %in, %in_3 : f32
                linalg.yield %15 : f32
              } -> tensor<64x64xf32>
              scf.yield %14 : tensor<64x64xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 32768 + d2 * 2097152 + d3 * 262144)>(%arg9, %arg12, %arg10, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
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
            %4 = tensor.empty() : tensor<64x64xf32>
            %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %6 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %7 = scf.for %arg13 = %c0 to %6 step %c1 iter_args(%arg14 = %5) -> (tensor<64x64xf32>) {
              %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 262144 + d2 * 32768)>(%arg13, %arg9, %arg12)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
              %10 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
              %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 512 + d2 * 64)>(%arg13, %arg10, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
              %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
              %13 = linalg.matmul ins(%10, %12 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %15 = arith.addf %in, %in_3 : f32
                linalg.yield %15 : f32
              } -> tensor<64x64xf32>
              scf.yield %14 : tensor<64x64xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg10, %arg11, %arg9, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
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
            %4 = tensor.empty() : tensor<64x64xf32>
            %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %6 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %7 = scf.for %arg13 = %c0 to %6 step %c1 iter_args(%arg14 = %5) -> (tensor<64x64xf32>) {
              %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 262144 + d2 * 32768)>(%arg13, %arg10, %arg12)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
              %10 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
              %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 512 + d2 * 64)>(%arg13, %arg9, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
              %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
              %13 = linalg.matmul ins(%10, %12 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %15 = arith.addf %in, %in_3 : f32
                linalg.yield %15 : f32
              } -> tensor<64x64xf32>
              scf.yield %14 : tensor<64x64xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg9, %arg11, %arg10, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
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
            %4 = tensor.empty() : tensor<64x64xf32>
            %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %6 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %7 = scf.for %arg13 = %c0 to %6 step %c1 iter_args(%arg14 = %5) -> (tensor<64x64xf32>) {
              %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 262144 + d2 * 32768)>(%arg13, %arg9, %arg12)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
              %10 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
              %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 512 + d2 * 64)>(%arg13, %arg10, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
              %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
              %13 = linalg.matmul ins(%10, %12 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %15 = arith.addf %in, %in_3 : f32
                linalg.yield %15 : f32
              } -> tensor<64x64xf32>
              scf.yield %14 : tensor<64x64xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg10, %arg11, %arg9, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
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
            %4 = tensor.empty() : tensor<64x64xf32>
            %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %6 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %7 = scf.for %arg13 = %c0 to %6 step %c1 iter_args(%arg14 = %5) -> (tensor<64x64xf32>) {
              %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 262144 + d2 * 32768)>(%arg13, %arg10, %arg12)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
              %10 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
              %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 512 + d2 * 64)>(%arg13, %arg9, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
              %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
              %13 = linalg.matmul ins(%10, %12 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %15 = arith.addf %in, %in_3 : f32
                linalg.yield %15 : f32
              } -> tensor<64x64xf32>
              scf.yield %14 : tensor<64x64xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg9, %arg11, %arg10, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
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
            %4 = tensor.empty() : tensor<64x64xf32>
            %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %6 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %7 = scf.for %arg13 = %c0 to %6 step %c1 iter_args(%arg14 = %5) -> (tensor<64x64xf32>) {
              %9 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 32)>(%arg13, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
              %10 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
              %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 64 + d2 * 4096 + d3 * 512)>(%arg13, %arg12, %arg10, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
              %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
              %13 = linalg.matmul ins(%10, %12 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %15 = arith.addf %in, %in_3 : f32
                linalg.yield %15 : f32
              } -> tensor<64x64xf32>
              scf.yield %14 : tensor<64x64xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 64 + d2 * 4096 + d3 * 512)>(%arg9, %arg12, %arg10, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
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
            %4 = tensor.empty() : tensor<64x64xf32>
            %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %6 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %7 = scf.for %arg13 = %c0 to %6 step %c1 iter_args(%arg14 = %5) -> (tensor<64x64xf32>) {
              %9 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 32)>(%arg13, %arg10)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
              %10 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
              %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 64 + d2 * 4096 + d3 * 512)>(%arg13, %arg12, %arg9, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
              %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
              %13 = linalg.matmul ins(%10, %12 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %15 = arith.addf %in, %in_3 : f32
                linalg.yield %15 : f32
              } -> tensor<64x64xf32>
              scf.yield %14 : tensor<64x64xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 64 + d2 * 4096 + d3 * 512)>(%arg10, %arg12, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
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
            %4 = tensor.empty() : tensor<64x64xf32>
            %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %6 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %7 = scf.for %arg13 = %c0 to %6 step %c1 iter_args(%arg14 = %5) -> (tensor<64x64xf32>) {
              %9 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 32)>(%arg13, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
              %10 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
              %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 64 + d2 * 4096 + d3 * 512)>(%arg13, %arg12, %arg10, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
              %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
              %13 = linalg.matmul ins(%10, %12 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %15 = arith.addf %in, %in_3 : f32
                linalg.yield %15 : f32
              } -> tensor<64x64xf32>
              scf.yield %14 : tensor<64x64xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 64 + d2 * 4096 + d3 * 512)>(%arg9, %arg12, %arg10, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
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
            %4 = tensor.empty() : tensor<64x64xf32>
            %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %6 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %7 = scf.for %arg13 = %c0 to %6 step %c1 iter_args(%arg14 = %5) -> (tensor<64x64xf32>) {
              %9 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 32)>(%arg13, %arg10)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() : memref<64x32xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
              %10 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
              %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 64 + d2 * 4096 + d3 * 512)>(%arg13, %arg12, %arg9, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
              %12 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
              %13 = linalg.matmul ins(%10, %12 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %15 = arith.addf %in, %in_3 : f32
                linalg.yield %15 : f32
              } -> tensor<64x64xf32>
              scf.yield %14 : tensor<64x64xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 64 + d2 * 4096 + d3 * 512)>(%arg10, %arg12, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
}
