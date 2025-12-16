module attributes {loom.block_k = 64 : index, loom.block_m = 64 : index, loom.block_n = 64 : index} {
  %0 = df.mat "FPU" {shape = [32, 32, 32]}
  %1 = df.vec "SFPU" {shape = [32]}
  %2 = df.spatial_dim "x", 8
  %3 = df.spatial_dim "y", 8
  %4 = df.core "core" {scaleout=(%2, %3) , scalein=(%0, %1, [8, 1])}
  %5 = df.memory "L1" {scaleout=(%2, %3) , size = 32768, bandwidth = 64}
  %6 = df.mux %4 : !df.compute, %5 : !df.memory  {map = affine_map<(d0, d1) -> (d0, d1)>}
  %7 = df.interconnects "horizontal_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>} : !df.interconnect
  %8 = df.interconnects "vertical_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 8)>} : !df.interconnect
  %9 = df.spatial_dim "d", 4
  %10 = df.memory "DRAM" {scaleout=(%9) , size = 34359738368, bandwidth = 512}
  %11 = df.interconnects %5 : !df.memory, %10 : !df.memory  {map = affine_map<(d0, d1) -> (d0 ceildiv 4 + (d1 ceildiv 4) * 2)>} : !df.interconnect
  func.func @matmul_kernel__d0i0_d1i0__f01(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) {
    affine.parallel (%arg12) = (0) to (8) {
      affine.parallel (%arg13) = (0) to (8) {
        affine.for %arg14 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg15 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg12, %arg13, %arg14)
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty(%arg9, %arg10) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %12, %arg9 : index
            %16 = arith.muli %arg15, %arg10 : index
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %17 = scf.for %arg16 = %c0 to %c8 step %c1 iter_args(%arg17 = %14) -> (tensor<?x?xf32>) {
              %19 = arith.muli %arg16, %arg11 : index
              %20 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%19, %15]
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [%arg9, %arg11], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc(%arg9, %arg11) : memref<?x?xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %21 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %22 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%16, %19]
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%22], sizes: [%arg11, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc(%arg11, %arg10) : memref<?x?xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %23 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %24 = linalg.matmul ins(%21, %23 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg17, %24 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg17 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %26 = arith.addf %in, %in_3 : f32
                linalg.yield %26 : f32
              } -> tensor<?x?xf32>
              scf.yield %25 : tensor<?x?xf32>
            }
            %18 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%16, %15]
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [%arg9, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %reinterpret_cast : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) {
    affine.parallel (%arg12) = (0) to (8) {
      affine.parallel (%arg13) = (0) to (8) {
        affine.for %arg14 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg15 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg12, %arg13, %arg15)
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty(%arg9, %arg10) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %12, %arg9 : index
            %16 = arith.muli %arg14, %arg10 : index
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %17 = scf.for %arg16 = %c0 to %c8 step %c1 iter_args(%arg17 = %14) -> (tensor<?x?xf32>) {
              %19 = arith.muli %arg16, %arg11 : index
              %20 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%19, %15]
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [%arg9, %arg11], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc(%arg9, %arg11) : memref<?x?xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %21 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %22 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%16, %19]
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%22], sizes: [%arg11, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc(%arg11, %arg10) : memref<?x?xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %23 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %24 = linalg.matmul ins(%21, %23 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg17, %24 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg17 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %26 = arith.addf %in, %in_3 : f32
                linalg.yield %26 : f32
              } -> tensor<?x?xf32>
              scf.yield %25 : tensor<?x?xf32>
            }
            %18 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%16, %15]
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [%arg9, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %reinterpret_cast : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) {
    affine.parallel (%arg12) = (0) to (8) {
      affine.parallel (%arg13) = (0) to (8) {
        affine.for %arg14 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg15 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg14)
            %cst = arith.constant 0.000000e+00 : f32
            %14 = tensor.empty(%arg9, %arg10) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %arg9 : index
            %17 = arith.muli %12, %arg10 : index
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %18 = scf.for %arg16 = %c0 to %c8 step %c1 iter_args(%arg17 = %15) -> (tensor<?x?xf32>) {
              %20 = arith.muli %arg16, %arg11 : index
              %21 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%20, %16]
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [%arg9, %arg11], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc(%arg9, %arg11) : memref<?x?xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %22 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %23 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%17, %20]
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%23], sizes: [%arg11, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc(%arg11, %arg10) : memref<?x?xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %24 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %25 = linalg.matmul ins(%22, %24 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg17, %25 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg17 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %27 = arith.addf %in, %in_3 : f32
                linalg.yield %27 : f32
              } -> tensor<?x?xf32>
              scf.yield %26 : tensor<?x?xf32>
            }
            %19 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%17, %16]
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [%arg9, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %reinterpret_cast : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) {
    affine.parallel (%arg12) = (0) to (8) {
      affine.parallel (%arg13) = (0) to (8) {
        affine.for %arg14 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg15 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg14)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            %cst = arith.constant 0.000000e+00 : f32
            %14 = tensor.empty(%arg9, %arg10) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %arg9 : index
            %17 = arith.muli %12, %arg10 : index
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %18 = scf.for %arg16 = %c0 to %c8 step %c1 iter_args(%arg17 = %15) -> (tensor<?x?xf32>) {
              %20 = arith.muli %arg16, %arg11 : index
              %21 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%20, %16]
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [%arg9, %arg11], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc(%arg9, %arg11) : memref<?x?xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %22 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %23 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%17, %20]
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%23], sizes: [%arg11, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc(%arg11, %arg10) : memref<?x?xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %24 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %25 = linalg.matmul ins(%22, %24 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg17, %25 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg17 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %27 = arith.addf %in, %in_3 : f32
                linalg.yield %27 : f32
              } -> tensor<?x?xf32>
              scf.yield %26 : tensor<?x?xf32>
            }
            %19 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%17, %16]
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [%arg9, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %reinterpret_cast : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) {
    affine.parallel (%arg12) = (0) to (8) {
      affine.parallel (%arg13) = (0) to (8) {
        affine.for %arg14 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg15 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg14)
            %cst = arith.constant 0.000000e+00 : f32
            %14 = tensor.empty(%arg9, %arg10) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %arg9 : index
            %17 = arith.muli %12, %arg10 : index
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %18 = scf.for %arg16 = %c0 to %c8 step %c1 iter_args(%arg17 = %15) -> (tensor<?x?xf32>) {
              %20 = arith.muli %arg16, %arg11 : index
              %21 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%20, %16]
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [%arg9, %arg11], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc(%arg9, %arg11) : memref<?x?xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %22 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %23 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%17, %20]
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%23], sizes: [%arg11, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc(%arg11, %arg10) : memref<?x?xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %24 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %25 = linalg.matmul ins(%22, %24 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg17, %25 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg17 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %27 = arith.addf %in, %in_3 : f32
                linalg.yield %27 : f32
              } -> tensor<?x?xf32>
              scf.yield %26 : tensor<?x?xf32>
            }
            %19 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%17, %16]
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [%arg9, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %reinterpret_cast : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) {
    affine.parallel (%arg12) = (0) to (8) {
      affine.parallel (%arg13) = (0) to (8) {
        affine.for %arg14 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg15 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg14)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            %cst = arith.constant 0.000000e+00 : f32
            %14 = tensor.empty(%arg9, %arg10) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %arg9 : index
            %17 = arith.muli %12, %arg10 : index
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %18 = scf.for %arg16 = %c0 to %c8 step %c1 iter_args(%arg17 = %15) -> (tensor<?x?xf32>) {
              %20 = arith.muli %arg16, %arg11 : index
              %21 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%20, %16]
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [%arg9, %arg11], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc(%arg9, %arg11) : memref<?x?xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %22 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %23 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%17, %20]
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%23], sizes: [%arg11, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc(%arg11, %arg10) : memref<?x?xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %24 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %25 = linalg.matmul ins(%22, %24 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg17, %25 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg17 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %27 = arith.addf %in, %in_3 : f32
                linalg.yield %27 : f32
              } -> tensor<?x?xf32>
              scf.yield %26 : tensor<?x?xf32>
            }
            %19 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%17, %16]
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [%arg9, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %reinterpret_cast : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) {
    affine.parallel (%arg12) = (0) to (8) {
      affine.parallel (%arg13) = (0) to (8) {
        affine.for %arg14 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg15 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg12, %arg13, %arg15)
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty(%arg9, %arg10) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %arg14, %arg9 : index
            %16 = arith.muli %12, %arg10 : index
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %17 = scf.for %arg16 = %c0 to %c8 step %c1 iter_args(%arg17 = %14) -> (tensor<?x?xf32>) {
              %19 = arith.muli %arg16, %arg11 : index
              %20 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%19, %15]
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [%arg9, %arg11], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc(%arg9, %arg11) : memref<?x?xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %21 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %22 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%16, %19]
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%22], sizes: [%arg11, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc(%arg11, %arg10) : memref<?x?xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %23 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %24 = linalg.matmul ins(%21, %23 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg17, %24 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg17 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %26 = arith.addf %in, %in_3 : f32
                linalg.yield %26 : f32
              } -> tensor<?x?xf32>
              scf.yield %25 : tensor<?x?xf32>
            }
            %18 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%16, %15]
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [%arg9, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %reinterpret_cast : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index) {
    affine.parallel (%arg12) = (0) to (8) {
      affine.parallel (%arg13) = (0) to (8) {
        affine.for %arg14 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg15 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg12, %arg13, %arg14)
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty(%arg9, %arg10) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %arg15, %arg9 : index
            %16 = arith.muli %12, %arg10 : index
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %17 = scf.for %arg16 = %c0 to %c8 step %c1 iter_args(%arg17 = %14) -> (tensor<?x?xf32>) {
              %19 = arith.muli %arg16, %arg11 : index
              %20 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%19, %15]
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [%arg9, %arg11], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc(%arg9, %arg11) : memref<?x?xf32>
              memref.copy %reinterpret_cast_0, %alloc : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %21 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %22 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%16, %19]
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%22], sizes: [%arg11, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc(%arg11, %arg10) : memref<?x?xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32>
              %23 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %24 = linalg.matmul ins(%21, %23 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg17, %24 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg17 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %26 = arith.addf %in, %in_3 : f32
                linalg.yield %26 : f32
              } -> tensor<?x?xf32>
              scf.yield %25 : tensor<?x?xf32>
            }
            %18 = affine.apply affine_map<()[s0, s1] -> (s1 * 512 + s0)>()[%16, %15]
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [%arg9, %arg10], strides: [512, 1] : memref<*xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %reinterpret_cast : (tensor<?x?xf32>, memref<?x?xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
}
