module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 4))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 4))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
                } {loom.iter_type = #loom.iter_type<temporal>}
              } {loom.iter_type = #loom.iter_type<temporal>}
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.for %arg6 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s2 * 4))>()[%20, %21, %22] {
              affine.for %arg7 = 0 to affine_map<()[s0, s1, s2] -> (4096 ceildiv (s1 * 8))>()[%20, %21, %22] {
                affine.for %arg8 = 0 to affine_map<()[s0, s1, s2] -> (8 ceildiv (s0 * 2))>()[%20, %21, %22] {
                  %24 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg6)
                  %27 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %28 = loom.semaphore_take %27 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %29 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %30 = loom.semaphore_take %29 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %31 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %32 = loom.semaphore_take %31 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %33 = loom.init_tensor %32[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %35 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %34) -> (tensor<?x?x?xf16>) {
                    %41 = arith.muli %24, %20 : index
                    %42 = arith.muli %25, %21 : index
                    %43 = arith.muli %arg9, %23 : index
                    %44 = loom.subview %arg0[%41, %42, %43] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %44, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %45 = loom.bufferize_to_tensor %30[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %46 = arith.muli %26, %22 : index
                    %47 = loom.subview %arg1[%41, %43, %46] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %47, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %48 = loom.bufferize_to_tensor %28[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %49 = linalg.batch_matmul ins(%45, %48 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %28 : memref<?x?x?xf16>
                    loom.semaphore_give %30 : memref<?x?x?xf16>
                    affine.yield %49 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %36 = arith.muli %24, %20 : index
                  %37 = arith.muli %25, %21 : index
                  %38 = arith.muli %26, %22 : index
                  %39 = loom.subview %arg2[%36, %37, %38] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %40 = loom.bufferize_to_memref %35 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %40, %39 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %32 : memref<?x?x?xf16>
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
