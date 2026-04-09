module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d1i0_d2i1_d0i2__f012(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d1i0_d2i1_d0i2__f021(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d1i0_d2i1_d0i2__f102(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d1i0_d2i1_d0i2__f120(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d1i0_d2i1_d0i2__f201(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d1i0_d2i1_d0i2__f210(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d2i0_d1i1_d0i2__f021(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d2i0_d1i1_d0i2__f102(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d2i0_d1i1_d0i2__f120(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d2i0_d1i1_d0i2__f201(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d2i0_d1i1_d0i2__f210(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d1i0_d2i1_d0i2__f012(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d1i0_d2i1_d0i2__f021(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d1i0_d2i1_d0i2__f102(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d1i0_d2i1_d0i2__f120(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d1i0_d2i1_d0i2__f201(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d1i0_d2i1_d0i2__f210(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d2i0_d1i1_d0i2__f021(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d2i0_d1i1_d0i2__f102(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d2i0_d1i1_d0i2__f120(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d2i0_d1i1_d0i2__f201(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d2i0_d1i1_d0i2__f210(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 8))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %arg3, %43 : index
                    %45 = loom.reduce_sum %41(UB : [%c0, %44], LB : [%c8, %44]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d1i0_d2i1_d0i2__f012(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 2))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %43, %c2 : index
                    %45 = loom.reduce_sum %41(UB : [%43, %arg3], LB : [%44, %arg3]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d1i0_d2i1_d0i2__f021(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 2))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %43, %c2 : index
                    %45 = loom.reduce_sum %41(UB : [%43, %arg3], LB : [%44, %arg3]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d1i0_d2i1_d0i2__f102(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 2))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %43, %c2 : index
                    %45 = loom.reduce_sum %41(UB : [%43, %arg3], LB : [%44, %arg3]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d1i0_d2i1_d0i2__f120(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 2))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %43, %c2 : index
                    %45 = loom.reduce_sum %41(UB : [%43, %arg3], LB : [%44, %arg3]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d1i0_d2i1_d0i2__f201(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 2))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %43, %c2 : index
                    %45 = loom.reduce_sum %41(UB : [%43, %arg3], LB : [%44, %arg3]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d1i0_d2i1_d0i2__f210(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 2))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c2 : index
                    %44 = arith.addi %43, %c2 : index
                    %45 = loom.reduce_sum %41(UB : [%43, %arg3], LB : [%44, %arg3]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 2))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.addi %arg3, %c8 : index
                    %44 = loom.reduce_sum %41(UB : [%arg3, %arg4], LB : [%43, %arg4]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %45 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %46 = loom.bufferize_to_memref %44 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d2i0_d1i1_d0i2__f021(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 2))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.addi %arg3, %c8 : index
                    %44 = loom.reduce_sum %41(UB : [%arg3, %arg4], LB : [%43, %arg4]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %45 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %46 = loom.bufferize_to_memref %44 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d2i0_d1i1_d0i2__f102(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 2))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.addi %arg3, %c8 : index
                    %44 = loom.reduce_sum %41(UB : [%arg3, %arg4], LB : [%43, %arg4]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %45 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %46 = loom.bufferize_to_memref %44 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d2i0_d1i1_d0i2__f120(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 2))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.addi %arg3, %c8 : index
                    %44 = loom.reduce_sum %41(UB : [%arg3, %arg4], LB : [%43, %arg4]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %45 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %46 = loom.bufferize_to_memref %44 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d2i0_d1i1_d0i2__f201(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 2))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.addi %arg3, %c8 : index
                    %44 = loom.reduce_sum %41(UB : [%arg3, %arg4], LB : [%43, %arg4]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %45 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %46 = loom.bufferize_to_memref %44 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d2i0_d1i1_d0i2__f210(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 2))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 4))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.addi %arg3, %c8 : index
                    %44 = loom.reduce_sum %41(UB : [%arg3, %arg4], LB : [%43, %arg4]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %45 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %46 = loom.bufferize_to_memref %44 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d1i0_d2i1_d0i2__f012(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 4))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %43, %c4 : index
                    %45 = loom.reduce_sum %41(UB : [%43, %arg3], LB : [%44, %arg3]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d1i0_d2i1_d0i2__f021(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 4))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %43, %c4 : index
                    %45 = loom.reduce_sum %41(UB : [%43, %arg3], LB : [%44, %arg3]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d1i0_d2i1_d0i2__f102(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 4))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %43, %c4 : index
                    %45 = loom.reduce_sum %41(UB : [%43, %arg3], LB : [%44, %arg3]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d1i0_d2i1_d0i2__f120(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 4))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %43, %c4 : index
                    %45 = loom.reduce_sum %41(UB : [%43, %arg3], LB : [%44, %arg3]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d1i0_d2i1_d0i2__f201(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 4))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %43, %c4 : index
                    %45 = loom.reduce_sum %41(UB : [%43, %arg3], LB : [%44, %arg3]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d1i0_d2i1_d0i2__f210(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 4))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.muli %arg4, %c4 : index
                    %44 = arith.addi %43, %c4 : index
                    %45 = loom.reduce_sum %41(UB : [%43, %arg3], LB : [%44, %arg3]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %46 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %47 = loom.bufferize_to_memref %45 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 4))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.addi %arg3, %c8 : index
                    %44 = loom.reduce_sum %41(UB : [%arg3, %arg4], LB : [%43, %arg4]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %45 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %46 = loom.bufferize_to_memref %44 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d2i0_d1i1_d0i2__f021(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 4))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.addi %arg3, %c8 : index
                    %44 = loom.reduce_sum %41(UB : [%arg3, %arg4], LB : [%43, %arg4]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %45 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %46 = loom.bufferize_to_memref %44 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d2i0_d1i1_d0i2__f102(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 4))>()[%22] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.addi %arg3, %c8 : index
                    %44 = loom.reduce_sum %41(UB : [%arg3, %arg4], LB : [%43, %arg4]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %45 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %46 = loom.bufferize_to_memref %44 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d2i0_d1i1_d0i2__f120(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%21] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 4))>()[%22] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.addi %arg3, %c8 : index
                    %44 = loom.reduce_sum %41(UB : [%arg3, %arg4], LB : [%43, %arg4]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %45 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %46 = loom.bufferize_to_memref %44 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d2i0_d1i1_d0i2__f201(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 4))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%21] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.addi %arg3, %c8 : index
                    %44 = loom.reduce_sum %41(UB : [%arg3, %arg4], LB : [%43, %arg4]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %45 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %46 = loom.bufferize_to_memref %44 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d2i0_d1i1_d0i2__f210(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv (s0 * 4))>()[%22] {
              affine.for %arg7 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 8))>()[%21] {
                affine.for %arg8 = 0 to affine_map<()[s0] -> (256 ceildiv (s0 * 2))>()[%20] {
                  %23 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg6)
                  %26 = arith.muli %23, %20 : index
                  %27 = arith.muli %24, %21 : index
                  %28 = arith.muli %25, %22 : index
                  %29 = loom.alloc [%20, %22] on @L1 : memref<?x?xf32>
                  %30 = loom.semaphore_take %29 : memref<?x?xf32> -> memref<?x?xf32>
                  %31 = loom.subview %arg1[%26, %28] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
                  loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
                  %32 = loom.bufferize_to_tensor %30[%20, %22] : memref<?x?xf32> -> tensor<?x?xf32>
                  %33 = loom.alloc [%22, %21] on @L1 : memref<?x?xf32>
                  %34 = loom.semaphore_take %33 : memref<?x?xf32> -> memref<?x?xf32>
                  %35 = loom.subview %arg2[%28, %27] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                  loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
                  %36 = loom.bufferize_to_tensor %34[%22, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %37 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                  %38 = loom.semaphore_take %37 : memref<?x?xf32> -> memref<?x?xf32>
                  %39 = loom.init_tensor %38[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                  %40 = linalg.fill ins(%cst : f32) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  %41 = linalg.matmul ins(%32, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
                  loom.semaphore_give %34 : memref<?x?xf32>
                  loom.semaphore_give %30 : memref<?x?xf32>
                  %42 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %42 {
                    %43 = arith.addi %arg3, %c8 : index
                    %44 = loom.reduce_sum %41(UB : [%arg3, %arg4], LB : [%43, %arg4]) : tensor<?x?xf32> -> tensor<?x?xf32>
                    %45 = loom.subview %arg0[%26, %27] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    %46 = loom.bufferize_to_memref %44 : tensor<?x?xf32> -> memref<?x?xf32>
                    loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %38 : memref<?x?xf32>
                  }
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
