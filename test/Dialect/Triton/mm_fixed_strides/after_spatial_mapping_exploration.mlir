#map = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map2 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map3 = affine_map<()[s0] -> ((s0 + 31) floordiv 32)>
#map4 = affine_map<(d0, d1) -> (d1 * 32768 + d0 * 32)>
#map5 = affine_map<(d0, d1) -> (d1 * 16384 + d0 * 64)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>
module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = "df.interconnects"(%0, %1) <{map = #map}> : (index, index) -> !df.interconnect
  %3 = "df.interconnects"(%0, %1) <{map = #map1}> : (index, index) -> !df.interconnect
  func.func @matmul_kernel__d0i0_d1i0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9, %arg10) = (0, 0) to ((%arg6 ceildiv 8) ceildiv 8, %arg7) {
      affine.parallel (%arg11) = (0) to (8) {
        %4 = affine.apply #map2(%arg9, %arg11)
        affine.parallel (%arg12) = (0) to (8) {
          %5 = affine.apply #map2(%4, %arg12)
          %cst = arith.constant 0.000000e+00 : f32
          %6 = tensor.empty() : tensor<64x64xf32>
          %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
          %8 = affine.apply #map3()[%arg5]
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %9 = scf.for %arg13 = %c0 to %8 step %c1 iter_args(%arg14 = %7) -> (tensor<64x64xf32>) {
            %11 = affine.apply #map4(%arg13, %5)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
            %alloc = memref.alloc() : memref<64x32xf32>
            memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
            %12 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
            %13 = affine.apply #map5(%arg10, %arg13)
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
            %alloc_2 = memref.alloc() : memref<32x64xf32>
            memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
            %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
            %15 = linalg.matmul ins(%12, %14 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %16 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%arg14, %15 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
            ^bb0(%in: f32, %in_3: f32, %out: f32):
              %17 = arith.addf %in, %in_3 : f32
              linalg.yield %17 : f32
            } -> tensor<64x64xf32>
            scf.yield %16 : tensor<64x64xf32>
          }
          %10 = affine.apply #map7(%arg10, %5)
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          bufferization.materialize_in_destination %9 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
        } {tmd.mapped_to = "x"}
      } {tmd.mapped_to = "y"}
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9, %arg10) = (0, 0) to ((%arg6 ceildiv 8) ceildiv 8, %arg7) {
      affine.parallel (%arg11) = (0) to (8) {
        %4 = affine.apply #map2(%arg9, %arg11)
        affine.parallel (%arg12) = (0) to (8) {
          %5 = affine.apply #map2(%4, %arg12)
          %cst = arith.constant 0.000000e+00 : f32
          %6 = tensor.empty() : tensor<64x64xf32>
          %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
          %8 = affine.apply #map3()[%arg5]
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %9 = scf.for %arg13 = %c0 to %8 step %c1 iter_args(%arg14 = %7) -> (tensor<64x64xf32>) {
            %11 = affine.apply #map4(%arg13, %5)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
            %alloc = memref.alloc() : memref<64x32xf32>
            memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
            %12 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
            %13 = affine.apply #map5(%arg10, %arg13)
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
            %alloc_2 = memref.alloc() : memref<32x64xf32>
            memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
            %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
            %15 = linalg.matmul ins(%12, %14 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %16 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%arg14, %15 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
            ^bb0(%in: f32, %in_3: f32, %out: f32):
              %17 = arith.addf %in, %in_3 : f32
              linalg.yield %17 : f32
            } -> tensor<64x64xf32>
            scf.yield %16 : tensor<64x64xf32>
          }
          %10 = affine.apply #map7(%arg10, %5)
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          bufferization.materialize_in_destination %9 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
        } {tmd.mapped_to = "y"}
      } {tmd.mapped_to = "x"}
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9, %arg10) = (0, 0) to (%arg6 ceildiv 8, %arg7 ceildiv 8) {
      affine.parallel (%arg11) = (0) to (8) {
        %4 = affine.apply #map2(%arg10, %arg11)
        affine.parallel (%arg12) = (0) to (8) {
          %5 = affine.apply #map2(%arg9, %arg12)
          %cst = arith.constant 0.000000e+00 : f32
          %6 = tensor.empty() : tensor<64x64xf32>
          %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
          %8 = affine.apply #map3()[%arg5]
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %9 = scf.for %arg13 = %c0 to %8 step %c1 iter_args(%arg14 = %7) -> (tensor<64x64xf32>) {
            %11 = affine.apply #map4(%arg13, %5)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
            %alloc = memref.alloc() : memref<64x32xf32>
            memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
            %12 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
            %13 = affine.apply #map5(%4, %arg13)
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
            %alloc_2 = memref.alloc() : memref<32x64xf32>
            memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
            %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
            %15 = linalg.matmul ins(%12, %14 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %16 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%arg14, %15 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
            ^bb0(%in: f32, %in_3: f32, %out: f32):
              %17 = arith.addf %in, %in_3 : f32
              linalg.yield %17 : f32
            } -> tensor<64x64xf32>
            scf.yield %16 : tensor<64x64xf32>
          }
          %10 = affine.apply #map7(%4, %5)
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          bufferization.materialize_in_destination %9 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
        } {tmd.mapped_to = "x"}
      } {tmd.mapped_to = "y"}
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9, %arg10) = (0, 0) to (%arg6 ceildiv 8, %arg7 ceildiv 8) {
      affine.parallel (%arg11) = (0) to (8) {
        %4 = affine.apply #map2(%arg10, %arg11)
        affine.parallel (%arg12) = (0) to (8) {
          %5 = affine.apply #map2(%arg9, %arg12)
          %cst = arith.constant 0.000000e+00 : f32
          %6 = tensor.empty() : tensor<64x64xf32>
          %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
          %8 = affine.apply #map3()[%arg5]
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %9 = scf.for %arg13 = %c0 to %8 step %c1 iter_args(%arg14 = %7) -> (tensor<64x64xf32>) {
            %11 = affine.apply #map4(%arg13, %5)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
            %alloc = memref.alloc() : memref<64x32xf32>
            memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
            %12 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
            %13 = affine.apply #map5(%4, %arg13)
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
            %alloc_2 = memref.alloc() : memref<32x64xf32>
            memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
            %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
            %15 = linalg.matmul ins(%12, %14 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %16 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%arg14, %15 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
            ^bb0(%in: f32, %in_3: f32, %out: f32):
              %17 = arith.addf %in, %in_3 : f32
              linalg.yield %17 : f32
            } -> tensor<64x64xf32>
            scf.yield %16 : tensor<64x64xf32>
          }
          %10 = affine.apply #map7(%4, %5)
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          bufferization.materialize_in_destination %9 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
        } {tmd.mapped_to = "y"}
      } {tmd.mapped_to = "x"}
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9, %arg10) = (0, 0) to (%arg6, (%arg7 ceildiv 8) ceildiv 8) {
      affine.parallel (%arg11) = (0) to (8) {
        %4 = affine.apply #map2(%arg10, %arg11)
        affine.parallel (%arg12) = (0) to (8) {
          %5 = affine.apply #map2(%4, %arg12)
          %cst = arith.constant 0.000000e+00 : f32
          %6 = tensor.empty() : tensor<64x64xf32>
          %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
          %8 = affine.apply #map3()[%arg5]
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %9 = scf.for %arg13 = %c0 to %8 step %c1 iter_args(%arg14 = %7) -> (tensor<64x64xf32>) {
            %11 = affine.apply #map4(%arg13, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
            %alloc = memref.alloc() : memref<64x32xf32>
            memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
            %12 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
            %13 = affine.apply #map5(%5, %arg13)
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
            %alloc_2 = memref.alloc() : memref<32x64xf32>
            memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
            %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
            %15 = linalg.matmul ins(%12, %14 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %16 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%arg14, %15 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
            ^bb0(%in: f32, %in_3: f32, %out: f32):
              %17 = arith.addf %in, %in_3 : f32
              linalg.yield %17 : f32
            } -> tensor<64x64xf32>
            scf.yield %16 : tensor<64x64xf32>
          }
          %10 = affine.apply #map7(%5, %arg9)
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          bufferization.materialize_in_destination %9 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
        } {tmd.mapped_to = "x"}
      } {tmd.mapped_to = "y"}
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9, %arg10) = (0, 0) to (%arg6, (%arg7 ceildiv 8) ceildiv 8) {
      affine.parallel (%arg11) = (0) to (8) {
        %4 = affine.apply #map2(%arg10, %arg11)
        affine.parallel (%arg12) = (0) to (8) {
          %5 = affine.apply #map2(%4, %arg12)
          %cst = arith.constant 0.000000e+00 : f32
          %6 = tensor.empty() : tensor<64x64xf32>
          %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
          %8 = affine.apply #map3()[%arg5]
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %9 = scf.for %arg13 = %c0 to %8 step %c1 iter_args(%arg14 = %7) -> (tensor<64x64xf32>) {
            %11 = affine.apply #map4(%arg13, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [64, 32], strides: [512, 1] : memref<*xf32> to memref<64x32xf32, strided<[512, 1], offset: ?>>
            %alloc = memref.alloc() : memref<64x32xf32>
            memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[512, 1], offset: ?>> to memref<64x32xf32>
            %12 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
            %13 = affine.apply #map5(%5, %arg13)
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%13], sizes: [32, 64], strides: [512, 1] : memref<*xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
            %alloc_2 = memref.alloc() : memref<32x64xf32>
            memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[512, 1], offset: ?>> to memref<32x64xf32>
            %14 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
            %15 = linalg.matmul ins(%12, %14 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %16 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%arg14, %15 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg14 : tensor<64x64xf32>) {
            ^bb0(%in: f32, %in_3: f32, %out: f32):
              %17 = arith.addf %in, %in_3 : f32
              linalg.yield %17 : f32
            } -> tensor<64x64xf32>
            scf.yield %16 : tensor<64x64xf32>
          }
          %10 = affine.apply #map7(%5, %arg9)
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [64, 64], strides: [512, 1] : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
          bufferization.materialize_in_destination %9 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
        } {tmd.mapped_to = "y"}
      } {tmd.mapped_to = "x"}
    }
    return
  }
}

