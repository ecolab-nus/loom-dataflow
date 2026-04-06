module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
  %0 = adl.memory.bank "mem_DRAM_bank", {bsize = 8192 : i64, nblk = 196608 : i64}
  %1 = adl.spatial_dim "dim_dram_channel", 8
  %2 = adl.memory.array "mem_DRAM", [%1] of %0
  %3 = adl.resource.exclusive "res_L1_torus_h"
  %4 = adl.resource.exclusive "res_L1_torus_v"
  %5 = adl.memory.bank "mem_bank", {bsize = 16 : i64, nblk = 5856 : i64}
  %6 = adl.spatial_dim "dim_nbank", 16
  %7 = adl.memory.array "mem_L1", [%6] of %5
  %8 = adl.resource.exclusive "res_matrix_lane"
  %9 = adl.resource.exclusive "res_vector_lane"
  %10 = adl.processor.compute @proc_matrix_lane, [(%7, %7)], with [%8]
  %11 = adl.processor.compute @proc_vector_lane, [(%7, %7)], with [%9]
  %12 = adl.arch.compose "arch_core", arch[%10, %11], mem[%7]
  %13 = adl.spatial_dim "dim_x", 8
  %14 = adl.spatial_dim "dim_y", 8
  %15 = adl.arch.scale "arch_mesh", [%13, %14] of %12
  %16 = adl.processor.dmover @proc_dram_l1_mover, [(%2, %7), (%7, %2)], with [%3, %4]
  %17 = adl.processor.dmover @proc_dram_l1_bcst_v, [(%2, %7), (%7, %2)], with [%4]
  %18 = adl.processor.dmover @proc_dram_l1_bcst_h, [(%2, %7), (%7, %2)], with [%3]
  %19 = adl.arch.compose "arch_system", arch[%15, %16, %17, %18], mem[%2]
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f01__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg6, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg6, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f01__n_dim_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg6, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg6, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f01__n_dim_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg6, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg6, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f01__n_a_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg6, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg6, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f10__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg5, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg5, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f10__n_dim_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg5, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg5, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f10__n_dim_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg5, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg5, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f10__n_a_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg5, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg5, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f01__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg6, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg6, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f01__n_dim_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg6, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg6, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f01__n_dim_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg6, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg6, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f01__n_a_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg6, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg6, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f10__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg5, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg5, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f10__n_dim_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg5, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg5, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f10__n_dim_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg5, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg5, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f10__n_a_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %23, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %arg5, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %23, %20 : index
              %34 = arith.muli %arg5, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f01__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f01__n_dim_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f01__dim_x_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f01__dim_x_dim_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f10__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f10__n_dim_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f10__dim_x_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f10__dim_x_dim_y_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f01__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f01__n_dim_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f01__dim_y_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f01__dim_y_dim_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f10__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f10__n_dim_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f10__dim_y_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f10__dim_y_dim_x_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %25 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
              %31 = loom.init_tensor %30[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %33 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %32) -> (tensor<?x?xf16>) {
                %38 = arith.muli %23, %20 : index
                %39 = arith.muli %arg7, %22 : index
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %43, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %44 = loom.bufferize_to_tensor %26[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %45 = linalg.matmul ins(%41, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                loom.semaphore_give %28 : memref<?x?xf16>
                affine.yield %45 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %34 = arith.muli %23, %20 : index
              %35 = arith.muli %24, %21 : index
              %36 = loom.subview %arg2[%34, %35] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %37 = loom.bufferize_to_memref %33 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %37, %36 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %30 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f01__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg5, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg5, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f01__dim_y_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg5, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg5, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f01__dim_x_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg5, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg5, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f01__a_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg5, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg5, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f10__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg6, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg6, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f10__dim_y_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg6, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg6, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f10__dim_x_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg6, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg6, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f10__a_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg6, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg6, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f01__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg5, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg5, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f01__dim_y_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg5, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg5, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f01__dim_x_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg5, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg5, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f01__a_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg5, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg5, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f10__n_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg6, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg6, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f10__dim_y_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg6, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg6, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f10__dim_x_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg6, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg6, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f10__a_n_n(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%20, %21] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%20, %21] {
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %24 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
              %30 = loom.init_tensor %29[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %32 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%22] iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
                %37 = arith.muli %arg6, %20 : index
                %38 = arith.muli %arg7, %22 : index
                %39 = loom.subview %arg0[%37, %38] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %39, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %40 = loom.bufferize_to_tensor %27[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = arith.muli %23, %21 : index
                %42 = loom.subview %arg1[%38, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %42, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %43 = loom.bufferize_to_tensor %25[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %44 = linalg.matmul ins(%40, %43 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                affine.yield %44 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %33 = arith.muli %arg6, %20 : index
              %34 = arith.muli %23, %21 : index
              %35 = loom.subview %arg2[%33, %34] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %29 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
}
