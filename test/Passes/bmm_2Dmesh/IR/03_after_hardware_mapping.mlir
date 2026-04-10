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
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %24 = arith.muli %20, %c4 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c2 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %24 = arith.muli %20, %c4 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c2 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %24 = arith.muli %21, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c4 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c2 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %24 = arith.muli %21, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c2 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c4 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %24 = arith.muli %22, %c2 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c4 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %24 = arith.muli %22, %c2 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c4 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %20, %c4 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c2 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %20, %c4 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c2 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %21, %c2 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c4 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %21, %c2 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c4 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %22, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c4 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c2 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %22, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c2 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c4 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %24 = arith.muli %20, %c2 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c4 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %24 = arith.muli %20, %c2 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c4 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %24 = arith.muli %21, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c2 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c4 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %24 = arith.muli %21, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c4 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c2 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %24 = arith.muli %22, %c4 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c2 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %24 = arith.muli %22, %c4 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c2 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %20, %c2 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c4 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %20, %c2 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c4 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %21, %c4 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c2 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %21, %c4 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c2 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %22, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c2 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c4 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %22, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c4 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c2 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %20, %c4 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c2 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %20, %c4 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c2 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %21, %c2 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c4 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %21, %c2 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c4 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %22, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c4 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c2 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %22, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c2 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c4 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %24 = arith.muli %20, %c4 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c2 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %24 = arith.muli %20, %c4 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c2 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %24 = arith.muli %21, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c4 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c2 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %24 = arith.muli %21, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c2 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c4 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %24 = arith.muli %22, %c2 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c4 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %24 = arith.muli %22, %c2 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c4 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %20, %c2 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c4 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %20, %c2 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c4 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %21, %c4 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c2 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %21, %c4 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c2 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %22, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c2 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c4 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %24 = arith.muli %22, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c4 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c2 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %24 = arith.muli %20, %c2 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c4 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f021(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %24 = arith.muli %20, %c2 : index
            %25 = arith.ceildivui %c8, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c4 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f102(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %24 = arith.muli %21, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c2 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %22, %c4 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f120(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %24 = arith.muli %21, %c8 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %22, %c4 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c2 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f201(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %24 = arith.muli %22, %c4 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %20, %c2 : index
              %27 = arith.ceildivui %c8, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %21, %c8 : index
                %29 = arith.ceildivui %c4096, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg8)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f210(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %20 = loom.sym @tile_b {upper_bound = 8 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %24 = arith.muli %22, %c4 : index
            %25 = arith.ceildivui %c4096, %24 : index
            scf.for %arg6 = %c0 to %25 step %c1 {
              %26 = arith.muli %21, %c8 : index
              %27 = arith.ceildivui %c4096, %26 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.muli %20, %c2 : index
                %29 = arith.ceildivui %c8, %28 : index
                scf.for %arg8 = %c0 to %29 step %c1 {
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg8)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %32 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg6)
                  %33 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                  %34 = loom.semaphore_take %33 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %35 = loom.init_tensor %34[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %37 = loom.alloc [%20, %21, %23] on @L1 : memref<?x?x?xf16>
                  %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %39 = loom.alloc [%20, %23, %22] on @L1 : memref<?x?x?xf16>
                  %40 = loom.semaphore_take %39 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %41 = affine.for %arg9 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%23] iter_args(%arg10 = %36) -> (tensor<?x?x?xf16>) {
                    %47 = arith.muli %30, %20 : index
                    %48 = arith.muli %31, %21 : index
                    %49 = arith.muli %arg9, %23 : index
                    %50 = loom.subview %arg0[%47, %48, %49] [%20, %21, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
                    loom.copy %50, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
                    %51 = loom.bufferize_to_tensor %38[%20, %21, %23] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %52 = arith.muli %32, %22 : index
                    %53 = loom.subview %arg1[%47, %49, %52] [%20, %23, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
                    loom.copy %53, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
                    %54 = loom.bufferize_to_tensor %40[%20, %23, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %55 = linalg.batch_matmul ins(%51, %54 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg10 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %40 : memref<?x?x?xf16>
                    loom.semaphore_give %38 : memref<?x?x?xf16>
                    affine.yield %55 : tensor<?x?x?xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  %42 = arith.muli %30, %20 : index
                  %43 = arith.muli %31, %21 : index
                  %44 = arith.muli %32, %22 : index
                  %45 = loom.subview %arg2[%42, %43, %44] [%20, %21, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %41 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
                  loom.copy %46, %45 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
                  loom.semaphore_give %34 : memref<?x?x?xf16>
                } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
