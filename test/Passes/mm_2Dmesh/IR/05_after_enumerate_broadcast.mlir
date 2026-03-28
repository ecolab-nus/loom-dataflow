module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
  %0 = adl.memory.bank "DRAM_bank", {bsize = 256 : i64, nblk = 8192 : i64}
  %1 = adl.spatial_dim "dram_channel", 8
  %2 = adl.memory.array "DRAM", [%1] of %0
  %3 = adl.memory.bank "bank", {bsize = 128 : i64, nblk = 1024 : i64}
  %4 = adl.spatial_dim "nbank", 16
  %5 = adl.memory.array "L1", [%4] of %3
  %6 = adl.processor.compute @matrix_lane, [(%5, %5)]
  %7 = adl.processor.compute @vector_lane, [(%5, %5)]
  %8 = adl.arch.compose @core, arch[%6, %7], mem[%5]
  %9 = adl.spatial_dim "x", 8
  %10 = adl.spatial_dim "y", 8
  %11 = adl.arch.scale "mesh", [%9, %10] of %8
  %12 = adl.processor.dmover "dram_l1_mover", [(%2, %5), (%5, %2)]
  %13 = adl.arch.compose @system, arch[%11, %12], mem[%2]
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f01__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg6, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg6, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f01__n_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg6, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg6, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f01__n_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg6, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg6, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f01__n_a_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg6, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg6, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f10__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg5, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg5, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f10__n_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg5, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg5, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f10__n_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg5, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg5, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f10__n_a_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg5, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg5, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f01__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg6, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg6, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f01__n_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg6, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg6, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f01__n_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg6, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg6, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f01__n_a_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg6, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg6, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f10__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg5, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg5, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f10__n_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg5, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg5, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f10__n_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg5, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg5, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f10__n_a_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %17, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %arg5, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %17, %14 : index
              %28 = arith.muli %arg5, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f01__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f01__n_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f01__x_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f01__x_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f10__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f10__n_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f10__x_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f10__x_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f01__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f01__n_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f01__y_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f01__y_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f10__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f10__n_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f10__y_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f10__y_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %19 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.init_tensor %24[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %27 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %26) -> (tensor<?x?xf16>) {
                %32 = arith.muli %17, %14 : index
                %33 = arith.muli %arg7, %16 : index
                %34 = loom.subview %arg0[%32, %33] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %34, %22 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %35 = loom.bufferize_to_tensor %22[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %36 = arith.muli %18, %15 : index
                %37 = loom.subview %arg1[%33, %36] [%16, %15] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %37, %20 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %20[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                loom.semaphore_give %22 : memref<?x?xf16>
                affine.yield %39 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %28 = arith.muli %17, %14 : index
              %29 = arith.muli %18, %15 : index
              %30 = loom.subview %arg2[%28, %29] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %31 = loom.bufferize_to_memref %27 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %31, %30 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %24 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f01__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg5, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg5, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f01__y_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg5, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg5, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f01__x_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg5, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg5, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f01__a_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg5, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg5, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f10__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg6, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg6, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f10__y_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg6, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg6, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f10__x_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg6, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg6, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f10__a_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg6, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg6, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f01__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg5, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg5, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f01__y_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg5, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg5, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f01__x_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg5, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg5, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f01__a_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg5, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg5, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f10__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg6, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg6, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f10__y_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg6, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg6, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f10__x_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg6, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg6, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f10__a_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %14 = loom.sym @block_size_0 : index
      %15 = loom.sym @block_size_1 : index
      %16 = loom.sym @block_size_2 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%14, %15] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%14, %15] {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %18 = loom.alloc [%16, %15] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%14, %16] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.alloc [%14, %15] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.init_tensor %23[%14, %15] : memref<?x?xf16> -> tensor<?x?xf16>
              %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %26 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%16] iter_args(%arg8 = %25) -> (tensor<?x?xf16>) {
                %31 = arith.muli %arg6, %14 : index
                %32 = arith.muli %arg7, %16 : index
                %33 = loom.subview %arg0[%31, %32] [%14, %16] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %33, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %34 = loom.bufferize_to_tensor %21[%14, %16] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = arith.muli %17, %15 : index
                %36 = loom.subview %arg1[%32, %35] [%16, %15] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %36, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %37 = loom.bufferize_to_tensor %19[%16, %15] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.matmul ins(%34, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                loom.semaphore_give %21 : memref<?x?xf16>
                affine.yield %38 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %27 = arith.muli %arg6, %14 : index
              %28 = arith.muli %17, %15 : index
              %29 = loom.subview %arg2[%27, %28] [%14, %15] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %30 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %30, %29 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %23 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
}
