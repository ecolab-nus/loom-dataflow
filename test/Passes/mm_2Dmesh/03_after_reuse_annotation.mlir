module attributes {loom.block_k = 64 : index, loom.block_m = 64 : index, loom.block_n = 64 : index} {
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
  func.func @matmul_kernel__d0i0_d1i0__f01(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %13 = loom.get_module_attribute "loom.block_m" : index
            %14 = loom.get_module_attribute "loom.block_n" : index
            %15 = loom.get_module_attribute "loom.block_k" : index
            %16 = arith.ceildivsi %c512_0, %15 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c512_1 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
            %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %19 = arith.muli %12, %13 : index
            %20 = arith.muli %arg12, %14 : index
            %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
              %25 = arith.muli %arg13, %15 : index
              %26 = arith.muli %19, %c512_1 : index
              %27 = arith.addi %26, %25 : index
              %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%13, %15) : memref<?x?xf32>
              loom.copy %28, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = arith.muli %25, %c512_1 : index
              %31 = arith.addi %30, %20 : index
              %32 = loom.reinterpret_cast %arg1 to offset : [%31], sizes : [%15, %14], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_2 = memref.alloc(%15, %14) : memref<?x?xf32>
              loom.copy %32, %alloc_2, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %33 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %34 = linalg.matmul ins(%29, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %36 = arith.addf %in, %in_3 : f32
                linalg.yield %36 : f32
              } -> tensor<?x?xf32>
              scf.yield %35 : tensor<?x?xf32>
            }
            %22 = arith.muli %19, %c512_1 : index
            %23 = arith.addi %22, %20 : index
            %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %13 = loom.get_module_attribute "loom.block_m" : index
            %14 = loom.get_module_attribute "loom.block_n" : index
            %15 = loom.get_module_attribute "loom.block_k" : index
            %16 = arith.ceildivsi %c512_0, %15 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c512_1 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
            %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %19 = arith.muli %12, %13 : index
            %20 = arith.muli %arg11, %14 : index
            %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
              %25 = arith.muli %arg13, %15 : index
              %26 = arith.muli %19, %c512_1 : index
              %27 = arith.addi %26, %25 : index
              %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%13, %15) : memref<?x?xf32>
              loom.copy %28, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = arith.muli %25, %c512_1 : index
              %31 = arith.addi %30, %20 : index
              %32 = loom.reinterpret_cast %arg1 to offset : [%31], sizes : [%15, %14], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_2 = memref.alloc(%15, %14) : memref<?x?xf32>
              loom.copy %32, %alloc_2, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %33 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %34 = linalg.matmul ins(%29, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %36 = arith.addf %in, %in_3 : f32
                linalg.yield %36 : f32
              } -> tensor<?x?xf32>
              scf.yield %35 : tensor<?x?xf32>
            }
            %22 = arith.muli %19, %c512_1 : index
            %23 = arith.addi %22, %20 : index
            %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %14 = loom.get_module_attribute "loom.block_m" : index
            %15 = loom.get_module_attribute "loom.block_n" : index
            %16 = loom.get_module_attribute "loom.block_k" : index
            %17 = arith.ceildivsi %c512_0, %16 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c512_1 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
            %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %20 = arith.muli %13, %14 : index
            %21 = arith.muli %12, %15 : index
            %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
              %26 = arith.muli %arg13, %16 : index
              %27 = arith.muli %20, %c512_1 : index
              %28 = arith.addi %27, %26 : index
              %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%14, %16) : memref<?x?xf32>
              loom.copy %29, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = arith.muli %26, %c512_1 : index
              %32 = arith.addi %31, %21 : index
              %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%16, %15], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_2 = memref.alloc(%16, %15) : memref<?x?xf32>
              loom.copy %33, %alloc_2, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %34 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %35 = linalg.matmul ins(%30, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %37 = arith.addf %in, %in_3 : f32
                linalg.yield %37 : f32
              } -> tensor<?x?xf32>
              scf.yield %36 : tensor<?x?xf32>
            }
            %23 = arith.muli %20, %c512_1 : index
            %24 = arith.addi %23, %21 : index
            %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %14 = loom.get_module_attribute "loom.block_m" : index
            %15 = loom.get_module_attribute "loom.block_n" : index
            %16 = loom.get_module_attribute "loom.block_k" : index
            %17 = arith.ceildivsi %c512_0, %16 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c512_1 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
            %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %20 = arith.muli %13, %14 : index
            %21 = arith.muli %12, %15 : index
            %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
              %26 = arith.muli %arg13, %16 : index
              %27 = arith.muli %20, %c512_1 : index
              %28 = arith.addi %27, %26 : index
              %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%14, %16) : memref<?x?xf32>
              loom.copy %29, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = arith.muli %26, %c512_1 : index
              %32 = arith.addi %31, %21 : index
              %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%16, %15], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_2 = memref.alloc(%16, %15) : memref<?x?xf32>
              loom.copy %33, %alloc_2, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %34 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %35 = linalg.matmul ins(%30, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %37 = arith.addf %in, %in_3 : f32
                linalg.yield %37 : f32
              } -> tensor<?x?xf32>
              scf.yield %36 : tensor<?x?xf32>
            }
            %23 = arith.muli %20, %c512_1 : index
            %24 = arith.addi %23, %21 : index
            %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %14 = loom.get_module_attribute "loom.block_m" : index
            %15 = loom.get_module_attribute "loom.block_n" : index
            %16 = loom.get_module_attribute "loom.block_k" : index
            %17 = arith.ceildivsi %c512_0, %16 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c512_1 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
            %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %20 = arith.muli %13, %14 : index
            %21 = arith.muli %12, %15 : index
            %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
              %26 = arith.muli %arg13, %16 : index
              %27 = arith.muli %20, %c512_1 : index
              %28 = arith.addi %27, %26 : index
              %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%14, %16) : memref<?x?xf32>
              loom.copy %29, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = arith.muli %26, %c512_1 : index
              %32 = arith.addi %31, %21 : index
              %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%16, %15], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_2 = memref.alloc(%16, %15) : memref<?x?xf32>
              loom.copy %33, %alloc_2, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %34 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %35 = linalg.matmul ins(%30, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %37 = arith.addf %in, %in_3 : f32
                linalg.yield %37 : f32
              } -> tensor<?x?xf32>
              scf.yield %36 : tensor<?x?xf32>
            }
            %23 = arith.muli %20, %c512_1 : index
            %24 = arith.addi %23, %21 : index
            %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
            %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %14 = loom.get_module_attribute "loom.block_m" : index
            %15 = loom.get_module_attribute "loom.block_n" : index
            %16 = loom.get_module_attribute "loom.block_k" : index
            %17 = arith.ceildivsi %c512_0, %16 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c512_1 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
            %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %20 = arith.muli %13, %14 : index
            %21 = arith.muli %12, %15 : index
            %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
              %26 = arith.muli %arg13, %16 : index
              %27 = arith.muli %20, %c512_1 : index
              %28 = arith.addi %27, %26 : index
              %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%14, %16) : memref<?x?xf32>
              loom.copy %29, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %30 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %31 = arith.muli %26, %c512_1 : index
              %32 = arith.addi %31, %21 : index
              %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%16, %15], strides : [%c512, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_2 = memref.alloc(%16, %15) : memref<?x?xf32>
              loom.copy %33, %alloc_2, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %34 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %35 = linalg.matmul ins(%30, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %37 = arith.addf %in, %in_3 : f32
                linalg.yield %37 : f32
              } -> tensor<?x?xf32>
              scf.yield %36 : tensor<?x?xf32>
            }
            %23 = arith.muli %20, %c512_1 : index
            %24 = arith.addi %23, %21 : index
            %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @x}
    } {loom.mapped_to = @y}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %13 = loom.get_module_attribute "loom.block_m" : index
            %14 = loom.get_module_attribute "loom.block_n" : index
            %15 = loom.get_module_attribute "loom.block_k" : index
            %16 = arith.ceildivsi %c512_0, %15 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c512_1 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
            %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %19 = arith.muli %arg11, %13 : index
            %20 = arith.muli %12, %14 : index
            %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
              %25 = arith.muli %arg13, %15 : index
              %26 = arith.muli %19, %c512_1 : index
              %27 = arith.addi %26, %25 : index
              %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%13, %15) : memref<?x?xf32>
              loom.copy %28, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = arith.muli %25, %c512_1 : index
              %31 = arith.addi %30, %20 : index
              %32 = loom.reinterpret_cast %arg1 to offset : [%31], sizes : [%15, %14], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_2 = memref.alloc(%15, %14) : memref<?x?xf32>
              loom.copy %32, %alloc_2, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %33 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %34 = linalg.matmul ins(%29, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %36 = arith.addf %in, %in_3 : f32
                linalg.yield %36 : f32
              } -> tensor<?x?xf32>
              scf.yield %35 : tensor<?x?xf32>
            }
            %22 = arith.muli %19, %c512_1 : index
            %23 = arith.addi %22, %20 : index
            %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9) = (0) to (8) {
      affine.parallel (%arg10) = (0) to (8) {
        affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
            %c512 = arith.constant 512 : index
            %c512_0 = arith.constant 512 : index
            %c1 = arith.constant 1 : index
            %13 = loom.get_module_attribute "loom.block_m" : index
            %14 = loom.get_module_attribute "loom.block_n" : index
            %15 = loom.get_module_attribute "loom.block_k" : index
            %16 = arith.ceildivsi %c512_0, %15 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c512_1 = arith.constant 512 : index
            %c0 = arith.constant 0 : index
            %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
            %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %19 = arith.muli %arg12, %13 : index
            %20 = arith.muli %12, %14 : index
            %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
              %25 = arith.muli %arg13, %15 : index
              %26 = arith.muli %19, %c512_1 : index
              %27 = arith.addi %26, %25 : index
              %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c512_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc = memref.alloc(%13, %15) : memref<?x?xf32>
              loom.copy %28, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %29 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %30 = arith.muli %25, %c512_1 : index
              %31 = arith.addi %30, %20 : index
              %32 = loom.reinterpret_cast %arg1 to offset : [%31], sizes : [%15, %14], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %alloc_2 = memref.alloc(%15, %14) : memref<?x?xf32>
              loom.copy %32, %alloc_2, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
              %33 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
              %34 = linalg.matmul ins(%29, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %36 = arith.addf %in, %in_3 : f32
                linalg.yield %36 : f32
              } -> tensor<?x?xf32>
              scf.yield %35 : tensor<?x?xf32>
            }
            %22 = arith.muli %19, %c512_1 : index
            %23 = arith.addi %22, %20 : index
            %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c512, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = @y}
    } {loom.mapped_to = @x}
    return
  }
}
