module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y2y4__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc2_n(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 512 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c512, %20 : index
      %24 = arith.ceildivui %c512, %21 : index
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
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %37 = arith.muli %arg4, %c2 : index
                  %38 = arith.addi %arg3, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  %44 = arith.addi %37, %c1 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %37], LR : [%arg5, %44]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %45 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %46 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                  %48 = loom.init_tensor %47[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                  %50 = loom.init_tensor %49[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %51 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%39, %45 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %53 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %54 = loom.semaphore_take %53 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %55 = loom.init_tensor %54[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %56 = loom.gather ins(%52 : tensor<?x?xf16>) outs(%55 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %38], LR : [%c7, %38]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %49 : memref<?x?xf16>
                  %57 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %57 {
                    %58 = linalg.fill ins(%cst : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%56 : tensor<?x?x?xf16>) outs(%58 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %62 = arith.addf %in, %out : f16
                      linalg.yield %62 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %54 : memref<?x?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %47 : memref<?x?xf16>
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y2y4__d0i2_d1i1_d2i0__f012__dim_y_level0_bc2_n_n(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 512 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c512, %20 : index
      %24 = arith.ceildivui %c512, %21 : index
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
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %37 = arith.muli %arg3, %c2 : index
                  %38 = arith.addi %37, %c1 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %37], LR : [%arg5, %38]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  %44 = arith.addi %arg4, %37 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %44], LR : [%arg5, %44]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %45 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %46 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                  %48 = loom.init_tensor %47[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                  %50 = loom.init_tensor %49[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %51 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%39, %45 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %53 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %54 = loom.semaphore_take %53 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %55 = loom.init_tensor %54[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %56 = loom.gather ins(%52 : tensor<?x?xf16>) outs(%55 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %44], LR : [%c7, %44]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %49 : memref<?x?xf16>
                  %57 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %57 {
                    %58 = linalg.fill ins(%cst : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%56 : tensor<?x?x?xf16>) outs(%58 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %62 = arith.addf %in, %out : f16
                      linalg.yield %62 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %54 : memref<?x?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %44], LR : [%arg5, %44]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %47 : memref<?x?xf16>
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc4_n(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 512 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c512, %20 : index
      %24 = arith.ceildivui %c512, %21 : index
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
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %37 = arith.muli %arg4, %c4 : index
                  %38 = arith.addi %arg3, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  %44 = arith.addi %37, %c3 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %37], LR : [%arg5, %44]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %45 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %46 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                  %48 = loom.init_tensor %47[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                  %50 = loom.init_tensor %49[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %51 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%39, %45 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %53 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %54 = loom.semaphore_take %53 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %55 = loom.init_tensor %54[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %56 = loom.gather ins(%52 : tensor<?x?xf16>) outs(%55 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %38], LR : [%c7, %38]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %49 : memref<?x?xf16>
                  %57 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %57 {
                    %58 = linalg.fill ins(%cst : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%56 : tensor<?x?x?xf16>) outs(%58 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %62 = arith.addf %in, %out : f16
                      linalg.yield %62 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %54 : memref<?x?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %47 : memref<?x?xf16>
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i1_d2i0__f012__dim_y_level0_bc4_n_n(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 512 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c512, %20 : index
      %24 = arith.ceildivui %c512, %21 : index
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
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %37 = arith.muli %arg3, %c4 : index
                  %38 = arith.addi %37, %c3 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %37], LR : [%arg5, %38]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  %44 = arith.addi %arg4, %37 : index
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %44], LR : [%arg5, %44]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %45 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %46 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                  %48 = loom.init_tensor %47[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                  %50 = loom.init_tensor %49[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %51 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%39, %45 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %53 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %54 = loom.semaphore_take %53 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %55 = loom.init_tensor %54[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %56 = loom.gather ins(%52 : tensor<?x?xf16>) outs(%55 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %44], LR : [%c7, %44]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %49 : memref<?x?xf16>
                  %57 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %57 {
                    %58 = linalg.fill ins(%cst : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%56 : tensor<?x?x?xf16>) outs(%58 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %62 = arith.addf %in, %out : f16
                      linalg.yield %62 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %54 : memref<?x?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %44], LR : [%arg5, %44]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %47 : memref<?x?xf16>
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc8_n(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 512 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c512, %20 : index
      %24 = arith.ceildivui %c512, %21 : index
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
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %37 = arith.muli %arg4, %c2 : index
                  %38 = arith.addi %arg5, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %arg3], LR : [%38, %arg3]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%38, %c0], LR : [%38, %c7]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %44 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %45 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %46 = loom.semaphore_take %45 : memref<?x?xf16> -> memref<?x?xf16>
                  %47 = loom.init_tensor %46[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %48 = loom.semaphore_take %45 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %54 = loom.init_tensor %53[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %55 = arith.addi %37, %c1 : index
                  %56 = loom.gather ins(%51 : tensor<?x?xf16>) outs(%54 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%37, %arg3], LR : [%55, %arg3]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %48 : memref<?x?xf16>
                  %57 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %57 {
                    %58 = linalg.fill ins(%cst : f16) outs(%47 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%56 : tensor<?x?x?xf16>) outs(%58 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %62 = arith.addf %in, %out : f16
                      linalg.yield %62 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %53 : memref<?x?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%38, %arg3], LR : [%38, %arg3]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %46 : memref<?x?xf16>
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i1_d2i0__f012__dim_y_level0_bc8_n_n(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 512 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c512, %20 : index
      %24 = arith.ceildivui %c512, %21 : index
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
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %37 = arith.muli %arg3, %c2 : index
                  %38 = arith.addi %arg5, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%38, %c0], LR : [%38, %c7]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %arg4], LR : [%38, %arg4]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %44 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %45 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %46 = loom.semaphore_take %45 : memref<?x?xf16> -> memref<?x?xf16>
                  %47 = loom.init_tensor %46[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %48 = loom.semaphore_take %45 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %54 = loom.init_tensor %53[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %55 = arith.addi %37, %c1 : index
                  %56 = loom.gather ins(%51 : tensor<?x?xf16>) outs(%54 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%37, %arg4], LR : [%55, %arg4]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %48 : memref<?x?xf16>
                  %57 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %57 {
                    %58 = linalg.fill ins(%cst : f16) outs(%47 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%56 : tensor<?x?x?xf16>) outs(%58 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %62 = arith.addf %in, %out : f16
                      linalg.yield %62 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %53 : memref<?x?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%38, %arg4], LR : [%38, %arg4]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %46 : memref<?x?xf16>
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc8_n(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 512 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c512, %20 : index
      %24 = arith.ceildivui %c512, %21 : index
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
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %37 = arith.muli %arg4, %c4 : index
                  %38 = arith.addi %arg5, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %arg3], LR : [%38, %arg3]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%38, %c0], LR : [%38, %c7]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %44 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %45 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %46 = loom.semaphore_take %45 : memref<?x?xf16> -> memref<?x?xf16>
                  %47 = loom.init_tensor %46[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %48 = loom.semaphore_take %45 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %54 = loom.init_tensor %53[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %55 = arith.addi %37, %c3 : index
                  %56 = loom.gather ins(%51 : tensor<?x?xf16>) outs(%54 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%37, %arg3], LR : [%55, %arg3]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %48 : memref<?x?xf16>
                  %57 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %57 {
                    %58 = linalg.fill ins(%cst : f16) outs(%47 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%56 : tensor<?x?x?xf16>) outs(%58 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %62 = arith.addf %in, %out : f16
                      linalg.yield %62 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %53 : memref<?x?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%38, %arg3], LR : [%38, %arg3]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %46 : memref<?x?xf16>
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i1_d2i0__f012__dim_y_level0_bc8_n_n(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %20 = loom.sym @tile_m {upper_bound = 512 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 512 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c512, %20 : index
      %24 = arith.ceildivui %c512, %21 : index
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
                  %36 = loom.subview %arg1[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %37 = arith.muli %arg3, %c4 : index
                  %38 = arith.addi %arg5, %37 : index
                  loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%38, %c0], LR : [%38, %c7]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = arith.muli %30, %21 : index
                  %41 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
                  %43 = loom.subview %arg2[%33, %40] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %arg4], LR : [%38, %arg4]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %44 = loom.bufferize_to_tensor %42[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %45 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %46 = loom.semaphore_take %45 : memref<?x?xf16> -> memref<?x?xf16>
                  %47 = loom.init_tensor %46[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %48 = loom.semaphore_take %45 : memref<?x?xf16> -> memref<?x?xf16>
                  %49 = loom.init_tensor %48[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %51 = linalg.matmul ins(%39, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %42 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %52 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %54 = loom.init_tensor %53[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %55 = arith.addi %37, %c3 : index
                  %56 = loom.gather ins(%51 : tensor<?x?xf16>) outs(%54 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%37, %arg4], LR : [%55, %arg4]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %48 : memref<?x?xf16>
                  %57 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %57 {
                    %58 = linalg.fill ins(%cst : f16) outs(%47 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%56 : tensor<?x?x?xf16>) outs(%58 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %62 = arith.addf %in, %out : f16
                      linalg.yield %62 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %53 : memref<?x?x?xf16>
                    %60 = loom.subview %arg0[%32, %40] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %61 = loom.bufferize_to_memref %59 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %61, %60 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%38, %arg4], LR : [%38, %arg4]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %46 : memref<?x?xf16>
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
