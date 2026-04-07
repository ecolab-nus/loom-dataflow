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
    func.func @_matmul__x8_y8__d0i0_d1i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
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
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
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
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
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
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
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
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
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
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
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
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
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
                %40 = loom.subview %arg0[%38, %39] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %40, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %28[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = arith.muli %24, %21 : index
                %43 = loom.subview %arg1[%39, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
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
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
