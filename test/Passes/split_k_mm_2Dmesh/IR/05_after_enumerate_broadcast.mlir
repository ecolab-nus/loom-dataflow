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
    func.func @split_k_matmul__x8_y2y4__d1i0_d2i1_d0i2__f012__n_n_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %26 = arith.ceildivui %23, %c2 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c4 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c8 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c2_0 = arith.constant 2 : index
                  %37 = arith.muli %arg4, %c2_0 : index
                  %38 = arith.addi %arg3, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c2_1 = arith.constant 2 : index
                  %44 = arith.muli %arg4, %c2_1 : index
                  %45 = arith.addi %arg3, %44 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %45], LR : [%arg5, %45]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %48 = loom.semaphore_take %47 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %55 {
                    %56 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %57 = arith.muli %arg4, %c2 : index
                    %58 = arith.addi %arg3, %57 : index
                    %59 = loom.reduce_sum ins(%51) outs(%56) region : (UL : [%c0, %58], LR : [%c7, %58]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %48 : memref<?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c2_2 = arith.constant 2 : index
                    %62 = arith.muli %arg4, %c2_2 : index
                    %63 = arith.addi %arg3, %62 : index
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %63], LR : [%arg5, %63]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %53 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d1i0_d2i1_d0i2__f012__n_dim_y_level0_bc2_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %26 = arith.ceildivui %23, %c2 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c4 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c8 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c2_0 = arith.constant 2 : index
                  %37 = arith.muli %arg4, %c2_0 : index
                  %38 = arith.addi %arg3, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c0_1 = arith.constant 0 : index
                  %c2_2 = arith.constant 2 : index
                  %44 = arith.muli %arg4, %c2_2 : index
                  %45 = arith.addi %c0_1, %44 : index
                  %c1_3 = arith.constant 1 : index
                  %c2_4 = arith.constant 2 : index
                  %46 = arith.muli %arg4, %c2_4 : index
                  %47 = arith.addi %c1_3, %46 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %45], LR : [%arg5, %47]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.fill ins(%cst : f16) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %53 = linalg.matmul ins(%39, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%52 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %54 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %55 = loom.semaphore_take %54 : memref<?x?xf16> -> memref<?x?xf16>
                  %56 = loom.init_tensor %55[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %57 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %57 {
                    %58 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = arith.muli %arg4, %c2 : index
                    %60 = arith.addi %arg3, %59 : index
                    %61 = loom.reduce_sum ins(%53) outs(%58) region : (UL : [%c0, %60], LR : [%c7, %60]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %50 : memref<?x?xf16>
                    %62 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %63 = loom.bufferize_to_memref %61 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c2_5 = arith.constant 2 : index
                    %64 = arith.muli %arg4, %c2_5 : index
                    %65 = arith.addi %arg3, %64 : index
                    loom.copy %63, %62 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %65], LR : [%arg5, %65]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %55 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012__n_n_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %26 = arith.ceildivui %23, %c4 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c2 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c8 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c2_0 = arith.constant 2 : index
                  %37 = arith.muli %arg3, %c2_0 : index
                  %38 = arith.addi %arg4, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c2_1 = arith.constant 2 : index
                  %44 = arith.muli %arg3, %c2_1 : index
                  %45 = arith.addi %arg4, %44 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %45], LR : [%arg5, %45]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %48 = loom.semaphore_take %47 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %55 {
                    %56 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %57 = arith.muli %arg4, %c4 : index
                    %58 = arith.addi %arg3, %57 : index
                    %59 = loom.reduce_sum ins(%51) outs(%56) region : (UL : [%c0, %58], LR : [%c7, %58]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %48 : memref<?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c2_2 = arith.constant 2 : index
                    %62 = arith.muli %arg3, %c2_2 : index
                    %63 = arith.addi %arg4, %62 : index
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %63], LR : [%arg5, %63]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %53 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012__dim_y_level0_bc2_n_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %26 = arith.ceildivui %23, %c4 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c2 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c8 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c0_0 = arith.constant 0 : index
                  %c2_1 = arith.constant 2 : index
                  %37 = arith.muli %arg3, %c2_1 : index
                  %38 = arith.addi %c0_0, %37 : index
                  %c1_2 = arith.constant 1 : index
                  %c2_3 = arith.constant 2 : index
                  %39 = arith.muli %arg3, %c2_3 : index
                  %40 = arith.addi %c1_2, %39 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %38], LR : [%arg5, %40]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %41 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %42 = arith.muli %30, %21 : index
                  %43 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %44 = loom.semaphore_take %43 : memref<?x?xf16> -> memref<?x?xf16>
                  %45 = loom.subview %arg2[%33, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c2_4 = arith.constant 2 : index
                  %46 = arith.muli %arg3, %c2_4 : index
                  %47 = arith.addi %arg4, %46 : index
                  loom.copy %45, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %47], LR : [%arg5, %47]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %44[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.fill ins(%cst : f16) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %53 = linalg.matmul ins(%41, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%52 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %44 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %54 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %55 = loom.semaphore_take %54 : memref<?x?xf16> -> memref<?x?xf16>
                  %56 = loom.init_tensor %55[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %57 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %57 {
                    %58 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = arith.muli %arg4, %c4 : index
                    %60 = arith.addi %arg3, %59 : index
                    %61 = loom.reduce_sum ins(%53) outs(%58) region : (UL : [%c0, %60], LR : [%c7, %60]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %50 : memref<?x?xf16>
                    %62 = loom.subview %arg0[%32, %42] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %63 = loom.bufferize_to_memref %61 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c2_5 = arith.constant 2 : index
                    %64 = arith.muli %arg3, %c2_5 : index
                    %65 = arith.addi %arg4, %64 : index
                    loom.copy %63, %62 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %65], LR : [%arg5, %65]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %55 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d1i0_d2i1_d0i2__f012__n_n_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %26 = arith.ceildivui %23, %c4 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c2 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c8 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c4_0 = arith.constant 4 : index
                  %37 = arith.muli %arg4, %c4_0 : index
                  %38 = arith.addi %arg3, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c4_1 = arith.constant 4 : index
                  %44 = arith.muli %arg4, %c4_1 : index
                  %45 = arith.addi %arg3, %44 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %45], LR : [%arg5, %45]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %48 = loom.semaphore_take %47 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %55 {
                    %56 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %57 = arith.muli %arg4, %c4 : index
                    %58 = arith.addi %arg3, %57 : index
                    %59 = loom.reduce_sum ins(%51) outs(%56) region : (UL : [%c0, %58], LR : [%c7, %58]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %48 : memref<?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c4_2 = arith.constant 4 : index
                    %62 = arith.muli %arg4, %c4_2 : index
                    %63 = arith.addi %arg3, %62 : index
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %63], LR : [%arg5, %63]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %53 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d1i0_d2i1_d0i2__f012__n_dim_y_level0_bc4_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %26 = arith.ceildivui %23, %c4 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c2 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c8 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c4_0 = arith.constant 4 : index
                  %37 = arith.muli %arg4, %c4_0 : index
                  %38 = arith.addi %arg3, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c0_1 = arith.constant 0 : index
                  %c4_2 = arith.constant 4 : index
                  %44 = arith.muli %arg4, %c4_2 : index
                  %45 = arith.addi %c0_1, %44 : index
                  %c3 = arith.constant 3 : index
                  %c4_3 = arith.constant 4 : index
                  %46 = arith.muli %arg4, %c4_3 : index
                  %47 = arith.addi %c3, %46 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %45], LR : [%arg5, %47]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.fill ins(%cst : f16) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %53 = linalg.matmul ins(%39, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%52 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %54 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %55 = loom.semaphore_take %54 : memref<?x?xf16> -> memref<?x?xf16>
                  %56 = loom.init_tensor %55[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %57 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %57 {
                    %58 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = arith.muli %arg4, %c4 : index
                    %60 = arith.addi %arg3, %59 : index
                    %61 = loom.reduce_sum ins(%53) outs(%58) region : (UL : [%c0, %60], LR : [%c7, %60]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %50 : memref<?x?xf16>
                    %62 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %63 = loom.bufferize_to_memref %61 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c4_4 = arith.constant 4 : index
                    %64 = arith.muli %arg4, %c4_4 : index
                    %65 = arith.addi %arg3, %64 : index
                    loom.copy %63, %62 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %65], LR : [%arg5, %65]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %55 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012__n_n_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %26 = arith.ceildivui %23, %c2 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c4 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c8 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c4_0 = arith.constant 4 : index
                  %37 = arith.muli %arg3, %c4_0 : index
                  %38 = arith.addi %arg4, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c4_1 = arith.constant 4 : index
                  %44 = arith.muli %arg3, %c4_1 : index
                  %45 = arith.addi %arg4, %44 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %45], LR : [%arg5, %45]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %48 = loom.semaphore_take %47 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %55 {
                    %56 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %57 = arith.muli %arg4, %c2 : index
                    %58 = arith.addi %arg3, %57 : index
                    %59 = loom.reduce_sum ins(%51) outs(%56) region : (UL : [%c0, %58], LR : [%c7, %58]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %48 : memref<?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c4_2 = arith.constant 4 : index
                    %62 = arith.muli %arg3, %c4_2 : index
                    %63 = arith.addi %arg4, %62 : index
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %63], LR : [%arg5, %63]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %53 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012__dim_y_level0_bc4_n_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %26 = arith.ceildivui %23, %c2 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c4 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c8 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c0_0 = arith.constant 0 : index
                  %c4_1 = arith.constant 4 : index
                  %37 = arith.muli %arg3, %c4_1 : index
                  %38 = arith.addi %c0_0, %37 : index
                  %c3 = arith.constant 3 : index
                  %c4_2 = arith.constant 4 : index
                  %39 = arith.muli %arg3, %c4_2 : index
                  %40 = arith.addi %c3, %39 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %38], LR : [%arg5, %40]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %41 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %42 = arith.muli %30, %21 : index
                  %43 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %44 = loom.semaphore_take %43 : memref<?x?xf16> -> memref<?x?xf16>
                  %45 = loom.subview %arg2[%33, %42] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c4_3 = arith.constant 4 : index
                  %46 = arith.muli %arg3, %c4_3 : index
                  %47 = arith.addi %arg4, %46 : index
                  loom.copy %45, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %47], LR : [%arg5, %47]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %44[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.fill ins(%cst : f16) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %53 = linalg.matmul ins(%41, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%52 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %44 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %54 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %55 = loom.semaphore_take %54 : memref<?x?xf16> -> memref<?x?xf16>
                  %56 = loom.init_tensor %55[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %57 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %57 {
                    %58 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = arith.muli %arg4, %c2 : index
                    %60 = arith.addi %arg3, %59 : index
                    %61 = loom.reduce_sum ins(%53) outs(%58) region : (UL : [%c0, %60], LR : [%c7, %60]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %50 : memref<?x?xf16>
                    %62 = loom.subview %arg0[%32, %42] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %63 = loom.bufferize_to_memref %61 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c4_4 = arith.constant 4 : index
                    %64 = arith.muli %arg3, %c4_4 : index
                    %65 = arith.addi %arg4, %64 : index
                    loom.copy %63, %62 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %65], LR : [%arg5, %65]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %55 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d1i0_d2i1_d0i2__f012__n_n_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            %26 = arith.ceildivui %23, %c8 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c4 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c2 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c2_0 = arith.constant 2 : index
                  %37 = arith.muli %arg4, %c2_0 : index
                  %38 = arith.addi %arg5, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %arg3], LR : [%38, %arg3]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c2_1 = arith.constant 2 : index
                  %44 = arith.muli %arg4, %c2_1 : index
                  %45 = arith.addi %arg5, %44 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%45, %arg3], LR : [%45, %arg3]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %48 = loom.semaphore_take %47 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %55 {
                    %56 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %57 = arith.muli %arg4, %c2 : index
                    %58 = arith.addi %57, %c1 : index
                    %59 = loom.reduce_sum ins(%51) outs(%56) region : (UL : [%57, %arg3], LR : [%58, %arg3]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %48 : memref<?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c2_2 = arith.constant 2 : index
                    %62 = arith.muli %arg4, %c2_2 : index
                    %63 = arith.addi %arg5, %62 : index
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%63, %arg3], LR : [%63, %arg3]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %53 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d1i0_d2i1_d0i2__f012__n_dim_y_level0_bc8_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            %26 = arith.ceildivui %23, %c8 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c4 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c2 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c2_0 = arith.constant 2 : index
                  %37 = arith.muli %arg4, %c2_0 : index
                  %38 = arith.addi %arg5, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %arg3], LR : [%38, %arg3]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c2_1 = arith.constant 2 : index
                  %44 = arith.muli %arg4, %c2_1 : index
                  %45 = arith.addi %arg5, %44 : index
                  %c0_2 = arith.constant 0 : index
                  %c7 = arith.constant 7 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%45, %c0_2], LR : [%45, %c7]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %48 = loom.semaphore_take %47 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %55 {
                    %56 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %57 = arith.muli %arg4, %c2 : index
                    %58 = arith.addi %57, %c1 : index
                    %59 = loom.reduce_sum ins(%51) outs(%56) region : (UL : [%57, %arg3], LR : [%58, %arg3]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %48 : memref<?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c2_3 = arith.constant 2 : index
                    %62 = arith.muli %arg4, %c2_3 : index
                    %63 = arith.addi %arg5, %62 : index
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%63, %arg3], LR : [%63, %arg3]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %53 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012__n_n_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %26 = arith.ceildivui %23, %c4 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c8 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c2 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c2_0 = arith.constant 2 : index
                  %37 = arith.muli %arg3, %c2_0 : index
                  %38 = arith.addi %arg5, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %arg4], LR : [%38, %arg4]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c2_1 = arith.constant 2 : index
                  %44 = arith.muli %arg3, %c2_1 : index
                  %45 = arith.addi %arg5, %44 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%45, %arg4], LR : [%45, %arg4]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %48 = loom.semaphore_take %47 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %55 {
                    %56 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %57 = arith.addi %arg3, %c4 : index
                    %58 = loom.reduce_sum ins(%51) outs(%56) region : (UL : [%arg3, %arg4], LR : [%57, %arg4]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %48 : memref<?x?xf16>
                    %59 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %60 = loom.bufferize_to_memref %58 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c2_2 = arith.constant 2 : index
                    %61 = arith.muli %arg3, %c2_2 : index
                    %62 = arith.addi %arg5, %61 : index
                    loom.copy %60, %59 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%62, %arg4], LR : [%62, %arg4]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %53 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012__dim_y_level0_bc8_n_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %26 = arith.ceildivui %23, %c4 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c8 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c2 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c2_0 = arith.constant 2 : index
                  %37 = arith.muli %arg3, %c2_0 : index
                  %38 = arith.addi %arg5, %37 : index
                  %c0_1 = arith.constant 0 : index
                  %c7 = arith.constant 7 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%38, %c0_1], LR : [%38, %c7]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c2_2 = arith.constant 2 : index
                  %44 = arith.muli %arg3, %c2_2 : index
                  %45 = arith.addi %arg5, %44 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%45, %arg4], LR : [%45, %arg4]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %48 = loom.semaphore_take %47 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %55 {
                    %56 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %57 = arith.addi %arg3, %c4 : index
                    %58 = loom.reduce_sum ins(%51) outs(%56) region : (UL : [%arg3, %arg4], LR : [%57, %arg4]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %48 : memref<?x?xf16>
                    %59 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %60 = loom.bufferize_to_memref %58 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c2_3 = arith.constant 2 : index
                    %61 = arith.muli %arg3, %c2_3 : index
                    %62 = arith.addi %arg5, %61 : index
                    loom.copy %60, %59 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%62, %arg4], LR : [%62, %arg4]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %53 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d1i0_d2i1_d0i2__f012__n_n_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            %26 = arith.ceildivui %23, %c8 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c2 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c4 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c4_0 = arith.constant 4 : index
                  %37 = arith.muli %arg4, %c4_0 : index
                  %38 = arith.addi %arg5, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %arg3], LR : [%38, %arg3]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c4_1 = arith.constant 4 : index
                  %44 = arith.muli %arg4, %c4_1 : index
                  %45 = arith.addi %arg5, %44 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%45, %arg3], LR : [%45, %arg3]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %48 = loom.semaphore_take %47 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %55 {
                    %56 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %57 = arith.muli %arg4, %c4 : index
                    %58 = arith.addi %57, %c3 : index
                    %59 = loom.reduce_sum ins(%51) outs(%56) region : (UL : [%57, %arg3], LR : [%58, %arg3]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %48 : memref<?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c4_2 = arith.constant 4 : index
                    %62 = arith.muli %arg4, %c4_2 : index
                    %63 = arith.addi %arg5, %62 : index
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%63, %arg3], LR : [%63, %arg3]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %53 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d1i0_d2i1_d0i2__f012__n_dim_y_level0_bc8_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            %26 = arith.ceildivui %23, %c8 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c2 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c4 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c4_0 = arith.constant 4 : index
                  %37 = arith.muli %arg4, %c4_0 : index
                  %38 = arith.addi %arg5, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %arg3], LR : [%38, %arg3]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c4_1 = arith.constant 4 : index
                  %44 = arith.muli %arg4, %c4_1 : index
                  %45 = arith.addi %arg5, %44 : index
                  %c0_2 = arith.constant 0 : index
                  %c7 = arith.constant 7 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%45, %c0_2], LR : [%45, %c7]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %48 = loom.semaphore_take %47 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %55 {
                    %56 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %57 = arith.muli %arg4, %c4 : index
                    %58 = arith.addi %57, %c3 : index
                    %59 = loom.reduce_sum ins(%51) outs(%56) region : (UL : [%57, %arg3], LR : [%58, %arg3]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %48 : memref<?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c4_3 = arith.constant 4 : index
                    %62 = arith.muli %arg4, %c4_3 : index
                    %63 = arith.addi %arg5, %62 : index
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%63, %arg3], LR : [%63, %arg3]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %53 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012__n_n_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c6 = arith.constant 6 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %26 = arith.ceildivui %23, %c2 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c8 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c4 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c4_0 = arith.constant 4 : index
                  %37 = arith.muli %arg3, %c4_0 : index
                  %38 = arith.addi %arg5, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %arg4], LR : [%38, %arg4]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c4_1 = arith.constant 4 : index
                  %44 = arith.muli %arg3, %c4_1 : index
                  %45 = arith.addi %arg5, %44 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%45, %arg4], LR : [%45, %arg4]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %48 = loom.semaphore_take %47 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %55 {
                    %56 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %57 = arith.addi %arg3, %c6 : index
                    %58 = loom.reduce_sum ins(%51) outs(%56) region : (UL : [%arg3, %arg4], LR : [%57, %arg4]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %48 : memref<?x?xf16>
                    %59 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %60 = loom.bufferize_to_memref %58 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c4_2 = arith.constant 4 : index
                    %61 = arith.muli %arg3, %c4_2 : index
                    %62 = arith.addi %arg5, %61 : index
                    loom.copy %60, %59 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%62, %arg4], LR : [%62, %arg4]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %53 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012__dim_y_level0_bc8_n_n(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c6 = arith.constant 6 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c256, %20 : index
      %24 = arith.ceildivui %c256, %21 : index
      %25 = arith.ceildivui %c4096, %22 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %26 = arith.ceildivui %23, %c2 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %24, %c8 : index
              scf.for %arg7 = %c0 to %27 step %c1 {
                %28 = arith.ceildivui %25, %c4 : index
                scf.for %arg8 = %c0 to %28 step %c1 {
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %32 = arith.muli %29, %20 : index
                  %33 = arith.muli %31, %22 : index
                  %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                  %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %c4_0 = arith.constant 4 : index
                  %37 = arith.muli %arg3, %c4_0 : index
                  %38 = arith.addi %arg5, %37 : index
                  %c0_1 = arith.constant 0 : index
                  %c7 = arith.constant 7 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%38, %c0_1], LR : [%38, %c7]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  %c4_2 = arith.constant 4 : index
                  %44 = arith.muli %arg3, %c4_2 : index
                  %45 = arith.addi %arg5, %44 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%45, %arg4], LR : [%45, %arg4]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %48 = loom.semaphore_take %47 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %46 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %55 {
                    %56 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %57 = arith.addi %arg3, %c6 : index
                    %58 = loom.reduce_sum ins(%51) outs(%56) region : (UL : [%arg3, %arg4], LR : [%57, %arg4]) : tensor<?x?xf16> -> tensor<?x?xf16>
                    loom.semaphore_give %48 : memref<?x?xf16>
                    %59 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    %60 = loom.bufferize_to_memref %58 : tensor<?x?xf16> -> memref<?x?xf16>
                    %c4_3 = arith.constant 4 : index
                    %61 = arith.muli %arg3, %c4_3 : index
                    %62 = arith.addi %arg5, %61 : index
                    loom.copy %60, %59 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%62, %arg4], LR : [%62, %arg4]) : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.semaphore_give %53 : memref<?x?xf16>
                  }
                } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
