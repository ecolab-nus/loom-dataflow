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
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %13 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %12, %c64 : index
            %16 = arith.muli %arg12, %c64_1 : index
            %17 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %14) -> (tensor<?x?xf32>) {
              %21 = arith.muli %arg13, %c64_2 : index
              %22 = arith.muli %15, %c512_3 : index
              %23 = arith.addi %22, %21 : index
              %24 = loom.reinterpret_cast %arg0 to offset : [%23], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %24, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %25 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %26 = arith.muli %21, %c512_3 : index
              %27 = arith.addi %26, %16 : index
              %28 = loom.reinterpret_cast %arg1 to offset : [%27], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %28, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %32 = arith.addf %in, %in_5 : f32
                linalg.yield %32 : f32
              } -> tensor<?x?xf32>
              scf.yield %31 : tensor<?x?xf32>
            }
            %18 = arith.muli %15, %c512_3 : index
            %19 = arith.addi %18, %16 : index
            %20 = loom.reinterpret_cast %arg2 to offset : [%19], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %20 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f01__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %13 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %12, %c64 : index
            %16 = arith.muli %arg12, %c64_1 : index
            %17 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %14) -> (tensor<?x?xf32>) {
              %21 = arith.muli %arg13, %c64_2 : index
              %22 = arith.muli %15, %c512_3 : index
              %23 = arith.addi %22, %21 : index
              %24 = loom.reinterpret_cast %arg0 to offset : [%23], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %24, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %25 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %26 = arith.muli %21, %c512_3 : index
              %27 = arith.addi %26, %16 : index
              %28 = loom.reinterpret_cast %arg1 to offset : [%27], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %28, %alloc_4, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %32 = arith.addf %in, %in_5 : f32
                linalg.yield %32 : f32
              } -> tensor<?x?xf32>
              scf.yield %31 : tensor<?x?xf32>
            }
            %18 = arith.muli %15, %c512_3 : index
            %19 = arith.addi %18, %16 : index
            %20 = loom.reinterpret_cast %arg2 to offset : [%19], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %20 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f01__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %13 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %12, %c64 : index
            %16 = arith.muli %arg12, %c64_1 : index
            %17 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %14) -> (tensor<?x?xf32>) {
              %21 = arith.muli %arg13, %c64_2 : index
              %22 = arith.muli %15, %c512_3 : index
              %23 = arith.addi %22, %21 : index
              %24 = loom.reinterpret_cast %arg0 to offset : [%23], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %24, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %25 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %26 = arith.muli %21, %c512_3 : index
              %27 = arith.addi %26, %16 : index
              %28 = loom.reinterpret_cast %arg1 to offset : [%27], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %28, %alloc_4, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %32 = arith.addf %in, %in_5 : f32
                linalg.yield %32 : f32
              } -> tensor<?x?xf32>
              scf.yield %31 : tensor<?x?xf32>
            }
            %18 = arith.muli %15, %c512_3 : index
            %19 = arith.addi %18, %16 : index
            %20 = loom.reinterpret_cast %arg2 to offset : [%19], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %20 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %13 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %12, %c64 : index
            %16 = arith.muli %arg11, %c64_1 : index
            %17 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %14) -> (tensor<?x?xf32>) {
              %21 = arith.muli %arg13, %c64_2 : index
              %22 = arith.muli %15, %c512_3 : index
              %23 = arith.addi %22, %21 : index
              %24 = loom.reinterpret_cast %arg0 to offset : [%23], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %24, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %25 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %26 = arith.muli %21, %c512_3 : index
              %27 = arith.addi %26, %16 : index
              %28 = loom.reinterpret_cast %arg1 to offset : [%27], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %28, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %32 = arith.addf %in, %in_5 : f32
                linalg.yield %32 : f32
              } -> tensor<?x?xf32>
              scf.yield %31 : tensor<?x?xf32>
            }
            %18 = arith.muli %15, %c512_3 : index
            %19 = arith.addi %18, %16 : index
            %20 = loom.reinterpret_cast %arg2 to offset : [%19], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %20 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %13 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %12, %c64 : index
            %16 = arith.muli %arg11, %c64_1 : index
            %17 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %14) -> (tensor<?x?xf32>) {
              %21 = arith.muli %arg13, %c64_2 : index
              %22 = arith.muli %15, %c512_3 : index
              %23 = arith.addi %22, %21 : index
              %24 = loom.reinterpret_cast %arg0 to offset : [%23], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %24, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %25 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %26 = arith.muli %21, %c512_3 : index
              %27 = arith.addi %26, %16 : index
              %28 = loom.reinterpret_cast %arg1 to offset : [%27], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %28, %alloc_4, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %32 = arith.addf %in, %in_5 : f32
                linalg.yield %32 : f32
              } -> tensor<?x?xf32>
              scf.yield %31 : tensor<?x?xf32>
            }
            %18 = arith.muli %15, %c512_3 : index
            %19 = arith.addi %18, %16 : index
            %20 = loom.reinterpret_cast %arg2 to offset : [%19], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %20 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %13 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %12, %c64 : index
            %16 = arith.muli %arg11, %c64_1 : index
            %17 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %14) -> (tensor<?x?xf32>) {
              %21 = arith.muli %arg13, %c64_2 : index
              %22 = arith.muli %15, %c512_3 : index
              %23 = arith.addi %22, %21 : index
              %24 = loom.reinterpret_cast %arg0 to offset : [%23], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %24, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %25 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %26 = arith.muli %21, %c512_3 : index
              %27 = arith.addi %26, %16 : index
              %28 = loom.reinterpret_cast %arg1 to offset : [%27], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %28, %alloc_4, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %32 = arith.addf %in, %in_5 : f32
                linalg.yield %32 : f32
              } -> tensor<?x?xf32>
              scf.yield %31 : tensor<?x?xf32>
            }
            %18 = arith.muli %15, %c512_3 : index
            %19 = arith.addi %18, %16 : index
            %20 = loom.reinterpret_cast %arg2 to offset : [%19], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %20 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__h_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__h_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__v_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__v_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %14 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %16 = arith.muli %13, %c64 : index
            %17 = arith.muli %12, %c64_1 : index
            %18 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %15) -> (tensor<?x?xf32>) {
              %22 = arith.muli %arg13, %c64_2 : index
              %23 = arith.muli %16, %c512_3 : index
              %24 = arith.addi %23, %22 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %25, %alloc, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %26 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %27 = arith.muli %22, %c512_3 : index
              %28 = arith.addi %27, %17 : index
              %29 = loom.reinterpret_cast %arg1 to offset : [%28], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %29, %alloc_4, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = linalg.matmul ins(%26, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %33 = arith.addf %in, %in_5 : f32
                linalg.yield %33 : f32
              } -> tensor<?x?xf32>
              scf.yield %32 : tensor<?x?xf32>
            }
            %19 = arith.muli %16, %c512_3 : index
            %20 = arith.addi %19, %17 : index
            %21 = loom.reinterpret_cast %arg2 to offset : [%20], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %18 in writable %21 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %13 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %arg11, %c64 : index
            %16 = arith.muli %12, %c64_1 : index
            %17 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %14) -> (tensor<?x?xf32>) {
              %21 = arith.muli %arg13, %c64_2 : index
              %22 = arith.muli %15, %c512_3 : index
              %23 = arith.addi %22, %21 : index
              %24 = loom.reinterpret_cast %arg0 to offset : [%23], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %24, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %25 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %26 = arith.muli %21, %c512_3 : index
              %27 = arith.addi %26, %16 : index
              %28 = loom.reinterpret_cast %arg1 to offset : [%27], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %28, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %32 = arith.addf %in, %in_5 : f32
                linalg.yield %32 : f32
              } -> tensor<?x?xf32>
              scf.yield %31 : tensor<?x?xf32>
            }
            %18 = arith.muli %15, %c512_3 : index
            %19 = arith.addi %18, %16 : index
            %20 = loom.reinterpret_cast %arg2 to offset : [%19], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %20 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %13 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %arg11, %c64 : index
            %16 = arith.muli %12, %c64_1 : index
            %17 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %14) -> (tensor<?x?xf32>) {
              %21 = arith.muli %arg13, %c64_2 : index
              %22 = arith.muli %15, %c512_3 : index
              %23 = arith.addi %22, %21 : index
              %24 = loom.reinterpret_cast %arg0 to offset : [%23], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %24, %alloc, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %25 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %26 = arith.muli %21, %c512_3 : index
              %27 = arith.addi %26, %16 : index
              %28 = loom.reinterpret_cast %arg1 to offset : [%27], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %28, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %32 = arith.addf %in, %in_5 : f32
                linalg.yield %32 : f32
              } -> tensor<?x?xf32>
              scf.yield %31 : tensor<?x?xf32>
            }
            %18 = arith.muli %15, %c512_3 : index
            %19 = arith.addi %18, %16 : index
            %20 = loom.reinterpret_cast %arg2 to offset : [%19], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %20 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %13 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %arg11, %c64 : index
            %16 = arith.muli %12, %c64_1 : index
            %17 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %14) -> (tensor<?x?xf32>) {
              %21 = arith.muli %arg13, %c64_2 : index
              %22 = arith.muli %15, %c512_3 : index
              %23 = arith.addi %22, %21 : index
              %24 = loom.reinterpret_cast %arg0 to offset : [%23], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %24, %alloc, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %25 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %26 = arith.muli %21, %c512_3 : index
              %27 = arith.addi %26, %16 : index
              %28 = loom.reinterpret_cast %arg1 to offset : [%27], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %28, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %32 = arith.addf %in, %in_5 : f32
                linalg.yield %32 : f32
              } -> tensor<?x?xf32>
              scf.yield %31 : tensor<?x?xf32>
            }
            %18 = arith.muli %15, %c512_3 : index
            %19 = arith.addi %18, %16 : index
            %20 = loom.reinterpret_cast %arg2 to offset : [%19], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %20 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %13 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %arg12, %c64 : index
            %16 = arith.muli %12, %c64_1 : index
            %17 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %14) -> (tensor<?x?xf32>) {
              %21 = arith.muli %arg13, %c64_2 : index
              %22 = arith.muli %15, %c512_3 : index
              %23 = arith.addi %22, %21 : index
              %24 = loom.reinterpret_cast %arg0 to offset : [%23], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %24, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %25 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %26 = arith.muli %21, %c512_3 : index
              %27 = arith.addi %26, %16 : index
              %28 = loom.reinterpret_cast %arg1 to offset : [%27], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %28, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %32 = arith.addf %in, %in_5 : f32
                linalg.yield %32 : f32
              } -> tensor<?x?xf32>
              scf.yield %31 : tensor<?x?xf32>
            }
            %18 = arith.muli %15, %c512_3 : index
            %19 = arith.addi %18, %16 : index
            %20 = loom.reinterpret_cast %arg2 to offset : [%19], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %20 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %13 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %arg12, %c64 : index
            %16 = arith.muli %12, %c64_1 : index
            %17 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %14) -> (tensor<?x?xf32>) {
              %21 = arith.muli %arg13, %c64_2 : index
              %22 = arith.muli %15, %c512_3 : index
              %23 = arith.addi %22, %21 : index
              %24 = loom.reinterpret_cast %arg0 to offset : [%23], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %24, %alloc, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %25 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %26 = arith.muli %21, %c512_3 : index
              %27 = arith.addi %26, %16 : index
              %28 = loom.reinterpret_cast %arg1 to offset : [%27], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %28, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %32 = arith.addf %in, %in_5 : f32
                linalg.yield %32 : f32
              } -> tensor<?x?xf32>
              scf.yield %31 : tensor<?x?xf32>
            }
            %18 = arith.muli %15, %c512_3 : index
            %19 = arith.addi %18, %16 : index
            %20 = loom.reinterpret_cast %arg2 to offset : [%19], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %20 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c64_1 = arith.constant 64 : index
            %c64_2 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8 = arith.constant 8 : index
            %c512_3 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %13 = tensor.empty(%c64, %c64_1) : tensor<?x?xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %15 = arith.muli %arg12, %c64 : index
            %16 = arith.muli %12, %c64_1 : index
            %17 = scf.for %arg13 = %c0 to %c8 step %c1 iter_args(%arg14 = %14) -> (tensor<?x?xf32>) {
              %21 = arith.muli %arg13, %c64_2 : index
              %22 = arith.muli %15, %c512_3 : index
              %23 = arith.addi %22, %21 : index
              %24 = loom.reinterpret_cast %arg0 to offset : [%23], sizes : [%c64, %c64_2], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%c64, %c64_2) : memref<?x?xf32>
              loom.copy %24, %alloc, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %25 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %26 = arith.muli %21, %c512_3 : index
              %27 = arith.addi %26, %16 : index
              %28 = loom.reinterpret_cast %arg1 to offset : [%27], sizes : [%c64_2, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_4 = memref.alloc(%c64_2, %c64_1) : memref<?x?xf32>
              loom.copy %28, %alloc_4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc_4 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_5: f32, %out: f32):
                %32 = arith.addf %in, %in_5 : f32
                linalg.yield %32 : f32
              } -> tensor<?x?xf32>
              scf.yield %31 : tensor<?x?xf32>
            }
            %18 = arith.muli %15, %c512_3 : index
            %19 = arith.addi %18, %16 : index
            %20 = loom.reinterpret_cast %arg2 to offset : [%19], sizes : [%c64, %c64_1], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %17 in writable %20 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
}
