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
    func.func @split_k_matmul_gather__x8_y2y4__d0i2_d1i0_d2i1__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %41 = arith.muli %30, %21 : index
                  %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %43 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %56 : memref<?x?xf16>
                  %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %66 = arith.muli %arg4, %c2 : index
                  %67 = arith.addi %arg3, %66 : index
                  %68 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %67], LR : [%c7, %67]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %54 : memref<?x?xf16>
                  %69 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %69 {
                    %70 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %71 = loom.sync ins(%68 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %64 : memref<?x?x?xf16>
                    %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%71 : tensor<?x?x?xf16>) outs(%70 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %76 = arith.addf %in, %out : f16
                      linalg.yield %76 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %62 : memref<?x?x?xf16>
                    %73 = loom.sync ins(%72 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %52 : memref<?x?xf16>
                    %74 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %75 = loom.bufferize_to_memref %73 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %75, %74 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %50 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x8_y2y4__d0i2_d1i1_d2i0__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %41 = arith.muli %30, %21 : index
                  %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %43 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %56 : memref<?x?xf16>
                  %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %66 = arith.muli %arg3, %c2 : index
                  %67 = arith.addi %arg4, %66 : index
                  %68 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %67], LR : [%c7, %67]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %54 : memref<?x?xf16>
                  %69 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %69 {
                    %70 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %71 = loom.sync ins(%68 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %64 : memref<?x?x?xf16>
                    %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%71 : tensor<?x?x?xf16>) outs(%70 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %76 = arith.addf %in, %out : f16
                      linalg.yield %76 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %62 : memref<?x?x?xf16>
                    %73 = loom.sync ins(%72 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %52 : memref<?x?xf16>
                    %74 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %75 = loom.bufferize_to_memref %73 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %75, %74 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %50 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i0_d2i1__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %41 = arith.muli %30, %21 : index
                  %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %43 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %56 : memref<?x?xf16>
                  %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %66 = arith.muli %arg4, %c4 : index
                  %67 = arith.addi %arg3, %66 : index
                  %68 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %67], LR : [%c7, %67]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %54 : memref<?x?xf16>
                  %69 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %69 {
                    %70 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %71 = loom.sync ins(%68 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %64 : memref<?x?x?xf16>
                    %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%71 : tensor<?x?x?xf16>) outs(%70 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %76 = arith.addf %in, %out : f16
                      linalg.yield %76 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %62 : memref<?x?x?xf16>
                    %73 = loom.sync ins(%72 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %52 : memref<?x?xf16>
                    %74 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %75 = loom.bufferize_to_memref %73 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %75, %74 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %50 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i1_d2i0__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %41 = arith.muli %30, %21 : index
                  %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %43 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %56 : memref<?x?xf16>
                  %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %66 = arith.muli %arg3, %c4 : index
                  %67 = arith.addi %arg4, %66 : index
                  %68 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %67], LR : [%c7, %67]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %54 : memref<?x?xf16>
                  %69 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %69 {
                    %70 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %71 = loom.sync ins(%68 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %64 : memref<?x?x?xf16>
                    %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%71 : tensor<?x?x?xf16>) outs(%70 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %76 = arith.addf %in, %out : f16
                      linalg.yield %76 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %62 : memref<?x?x?xf16>
                    %73 = loom.sync ins(%72 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %52 : memref<?x?xf16>
                    %74 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %75 = loom.bufferize_to_memref %73 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %75, %74 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %50 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i0_d2i1__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %41 = arith.muli %30, %21 : index
                  %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %43 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %56 : memref<?x?xf16>
                  %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %66 = arith.muli %arg4, %c2 : index
                  %67 = arith.addi %66, %c1 : index
                  %68 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%66, %arg3], LR : [%67, %arg3]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %54 : memref<?x?xf16>
                  %69 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %69 {
                    %70 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %71 = loom.sync ins(%68 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %64 : memref<?x?x?xf16>
                    %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%71 : tensor<?x?x?xf16>) outs(%70 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %76 = arith.addf %in, %out : f16
                      linalg.yield %76 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %62 : memref<?x?x?xf16>
                    %73 = loom.sync ins(%72 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %52 : memref<?x?xf16>
                    %74 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %75 = loom.bufferize_to_memref %73 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %75, %74 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %50 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i1_d2i0__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %41 = arith.muli %30, %21 : index
                  %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %43 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %56 : memref<?x?xf16>
                  %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %66 = arith.muli %arg3, %c2 : index
                  %67 = arith.addi %66, %c1 : index
                  %68 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%66, %arg4], LR : [%67, %arg4]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %54 : memref<?x?xf16>
                  %69 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %69 {
                    %70 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %71 = loom.sync ins(%68 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %64 : memref<?x?x?xf16>
                    %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%71 : tensor<?x?x?xf16>) outs(%70 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %76 = arith.addf %in, %out : f16
                      linalg.yield %76 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %62 : memref<?x?x?xf16>
                    %73 = loom.sync ins(%72 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %52 : memref<?x?xf16>
                    %74 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %75 = loom.bufferize_to_memref %73 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %75, %74 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %50 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i0_d2i1__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %41 = arith.muli %30, %21 : index
                  %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %43 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %56 : memref<?x?xf16>
                  %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %66 = arith.muli %arg4, %c4 : index
                  %67 = arith.addi %66, %c3 : index
                  %68 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%66, %arg3], LR : [%67, %arg3]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %54 : memref<?x?xf16>
                  %69 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %69 {
                    %70 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %71 = loom.sync ins(%68 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %64 : memref<?x?x?xf16>
                    %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%71 : tensor<?x?x?xf16>) outs(%70 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %76 = arith.addf %in, %out : f16
                      linalg.yield %76 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %62 : memref<?x?x?xf16>
                    %73 = loom.sync ins(%72 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %52 : memref<?x?xf16>
                    %74 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %75 = loom.bufferize_to_memref %73 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %75, %74 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %50 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i1_d2i0__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %41 = arith.muli %30, %21 : index
                  %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                  %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %43 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %56 : memref<?x?xf16>
                  %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %66 = arith.muli %arg3, %c4 : index
                  %67 = arith.addi %66, %c3 : index
                  %68 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%66, %arg4], LR : [%67, %arg4]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %54 : memref<?x?xf16>
                  %69 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %69 {
                    %70 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %71 = loom.sync ins(%68 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %64 : memref<?x?x?xf16>
                    %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%71 : tensor<?x?x?xf16>) outs(%70 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %76 = arith.addf %in, %out : f16
                      linalg.yield %76 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %62 : memref<?x?x?xf16>
                    %73 = loom.sync ins(%72 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %52 : memref<?x?xf16>
                    %74 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %75 = loom.bufferize_to_memref %73 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %75, %74 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %50 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x8_y2y2y2__d0i2_d1i2_d2i0_d3i1__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
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
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.parallel (%arg6) = (0) to (2) {
              %26 = arith.ceildivui %23, %c2 : index
              scf.for %arg7 = %c0 to %26 step %c1 {
                %27 = arith.ceildivui %24, %c2 : index
                scf.for %arg8 = %c0 to %27 step %c1 {
                  %28 = arith.ceildivui %25, %c16 : index
                  scf.for %arg9 = %c0 to %28 step %c1 {
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                    %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                    %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg5, %arg6, %arg9)
                    %32 = arith.muli %29, %20 : index
                    %33 = arith.muli %31, %22 : index
                    %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                    %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %41 = arith.muli %30, %21 : index
                    %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %45 : memref<?x?xf16>
                    %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %43 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %56 : memref<?x?xf16>
                    %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %66 = arith.muli %arg3, %c2 : index
                    %67 = arith.addi %arg6, %66 : index
                    %68 = arith.muli %arg4, %c4 : index
                    %69 = arith.addi %67, %68 : index
                    %70 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %69], LR : [%c7, %69]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %71 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %71 {
                      %72 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %73 = loom.sync ins(%70 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %64 : memref<?x?x?xf16>
                      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%73 : tensor<?x?x?xf16>) outs(%72 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %78 = arith.addf %in, %out : f16
                        linalg.yield %78 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %62 : memref<?x?x?xf16>
                      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %52 : memref<?x?xf16>
                      %76 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %50 : memref<?x?xf16>
                    }
                  } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y2y2y2__d0i2_d1i2_d2i1_d3i0__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
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
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.parallel (%arg6) = (0) to (2) {
              %26 = arith.ceildivui %23, %c2 : index
              scf.for %arg7 = %c0 to %26 step %c1 {
                %27 = arith.ceildivui %24, %c2 : index
                scf.for %arg8 = %c0 to %27 step %c1 {
                  %28 = arith.ceildivui %25, %c16 : index
                  scf.for %arg9 = %c0 to %28 step %c1 {
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                    %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                    %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg5, %arg6, %arg9)
                    %32 = arith.muli %29, %20 : index
                    %33 = arith.muli %31, %22 : index
                    %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                    %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %41 = arith.muli %30, %21 : index
                    %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %45 : memref<?x?xf16>
                    %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %43 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %56 : memref<?x?xf16>
                    %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %66 = arith.muli %arg4, %c2 : index
                    %67 = arith.addi %arg6, %66 : index
                    %68 = arith.muli %arg3, %c4 : index
                    %69 = arith.addi %67, %68 : index
                    %70 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %69], LR : [%c7, %69]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %71 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %71 {
                      %72 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %73 = loom.sync ins(%70 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %64 : memref<?x?x?xf16>
                      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%73 : tensor<?x?x?xf16>) outs(%72 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %78 = arith.addf %in, %out : f16
                        linalg.yield %78 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %62 : memref<?x?x?xf16>
                      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %52 : memref<?x?xf16>
                      %76 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %50 : memref<?x?xf16>
                    }
                  } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y2y4__d0i2_d1i2_d2i0_d3i1__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.parallel (%arg6) = (0) to (2) {
              %26 = arith.ceildivui %23, %c4 : index
              scf.for %arg7 = %c0 to %26 step %c1 {
                %27 = arith.ceildivui %24, %c4 : index
                scf.for %arg8 = %c0 to %27 step %c1 {
                  %28 = arith.ceildivui %25, %c4 : index
                  scf.for %arg9 = %c0 to %28 step %c1 {
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                    %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                    %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 4)>(%arg5, %arg6, %arg9)
                    %32 = arith.muli %29, %20 : index
                    %33 = arith.muli %31, %22 : index
                    %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                    %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %41 = arith.muli %30, %21 : index
                    %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %45 : memref<?x?xf16>
                    %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %43 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %56 : memref<?x?xf16>
                    %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %66 = arith.muli %arg3, %c2 : index
                    %67 = arith.muli %arg4, %c2 : index
                    %68 = arith.addi %arg6, %67 : index
                    %69 = arith.addi %66, %c1 : index
                    %70 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%66, %68], LR : [%69, %68]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %71 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %71 {
                      %72 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %73 = loom.sync ins(%70 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %64 : memref<?x?x?xf16>
                      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%73 : tensor<?x?x?xf16>) outs(%72 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %78 = arith.addf %in, %out : f16
                        linalg.yield %78 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %62 : memref<?x?x?xf16>
                      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %52 : memref<?x?xf16>
                      %76 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %50 : memref<?x?xf16>
                    }
                  } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y2y4__d0i2_d1i2_d2i1_d3i0__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.parallel (%arg6) = (0) to (2) {
              %26 = arith.ceildivui %23, %c4 : index
              scf.for %arg7 = %c0 to %26 step %c1 {
                %27 = arith.ceildivui %24, %c4 : index
                scf.for %arg8 = %c0 to %27 step %c1 {
                  %28 = arith.ceildivui %25, %c4 : index
                  scf.for %arg9 = %c0 to %28 step %c1 {
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                    %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                    %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 4)>(%arg5, %arg6, %arg9)
                    %32 = arith.muli %29, %20 : index
                    %33 = arith.muli %31, %22 : index
                    %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                    %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %41 = arith.muli %30, %21 : index
                    %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %45 : memref<?x?xf16>
                    %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %43 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %56 : memref<?x?xf16>
                    %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %66 = arith.muli %arg4, %c2 : index
                    %67 = arith.muli %arg3, %c2 : index
                    %68 = arith.addi %arg6, %67 : index
                    %69 = arith.addi %66, %c1 : index
                    %70 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%66, %68], LR : [%69, %68]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %71 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %71 {
                      %72 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %73 = loom.sync ins(%70 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %64 : memref<?x?x?xf16>
                      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%73 : tensor<?x?x?xf16>) outs(%72 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %78 = arith.addf %in, %out : f16
                        linalg.yield %78 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %62 : memref<?x?x?xf16>
                      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %52 : memref<?x?xf16>
                      %76 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %50 : memref<?x?xf16>
                    }
                  } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y4y2__d0i2_d1i2_d2i0_d3i1__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
          affine.parallel (%arg5) = (0) to (2) {
            affine.parallel (%arg6) = (0) to (4) {
              %26 = arith.ceildivui %23, %c4 : index
              scf.for %arg7 = %c0 to %26 step %c1 {
                %27 = arith.ceildivui %24, %c2 : index
                scf.for %arg8 = %c0 to %27 step %c1 {
                  %28 = arith.ceildivui %25, %c8 : index
                  scf.for %arg9 = %c0 to %28 step %c1 {
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                    %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                    %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 8)>(%arg5, %arg6, %arg9)
                    %32 = arith.muli %29, %20 : index
                    %33 = arith.muli %31, %22 : index
                    %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                    %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %41 = arith.muli %30, %21 : index
                    %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %45 : memref<?x?xf16>
                    %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %43 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %56 : memref<?x?xf16>
                    %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %66 = arith.muli %arg3, %c2 : index
                    %67 = arith.muli %arg4, %c4 : index
                    %68 = arith.addi %arg6, %67 : index
                    %69 = arith.addi %66, %c1 : index
                    %70 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%66, %68], LR : [%69, %68]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %71 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %71 {
                      %72 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %73 = loom.sync ins(%70 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %64 : memref<?x?x?xf16>
                      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%73 : tensor<?x?x?xf16>) outs(%72 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %78 = arith.addf %in, %out : f16
                        linalg.yield %78 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %62 : memref<?x?x?xf16>
                      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %52 : memref<?x?xf16>
                      %76 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %50 : memref<?x?xf16>
                    }
                  } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y4y2__d0i2_d1i2_d2i1_d3i0__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
          affine.parallel (%arg5) = (0) to (2) {
            affine.parallel (%arg6) = (0) to (4) {
              %26 = arith.ceildivui %23, %c2 : index
              scf.for %arg7 = %c0 to %26 step %c1 {
                %27 = arith.ceildivui %24, %c4 : index
                scf.for %arg8 = %c0 to %27 step %c1 {
                  %28 = arith.ceildivui %25, %c8 : index
                  scf.for %arg9 = %c0 to %28 step %c1 {
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                    %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                    %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 8)>(%arg5, %arg6, %arg9)
                    %32 = arith.muli %29, %20 : index
                    %33 = arith.muli %31, %22 : index
                    %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                    %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %41 = arith.muli %30, %21 : index
                    %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %45 : memref<?x?xf16>
                    %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %43 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %56 : memref<?x?xf16>
                    %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %66 = arith.muli %arg4, %c2 : index
                    %67 = arith.muli %arg3, %c4 : index
                    %68 = arith.addi %arg6, %67 : index
                    %69 = arith.addi %66, %c1 : index
                    %70 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%66, %68], LR : [%69, %68]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %71 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %71 {
                      %72 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %73 = loom.sync ins(%70 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %64 : memref<?x?x?xf16>
                      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%73 : tensor<?x?x?xf16>) outs(%72 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %78 = arith.addf %in, %out : f16
                        linalg.yield %78 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %62 : memref<?x?x?xf16>
                      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %52 : memref<?x?xf16>
                      %76 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %50 : memref<?x?xf16>
                    }
                  } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y2y4__d0i2_d1i2_d2i0_d3i1__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c3 = arith.constant 3 : index
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
          affine.parallel (%arg5) = (0) to (4) {
            affine.parallel (%arg6) = (0) to (2) {
              %26 = arith.ceildivui %23, %c2 : index
              scf.for %arg7 = %c0 to %26 step %c1 {
                %27 = arith.ceildivui %24, %c4 : index
                scf.for %arg8 = %c0 to %27 step %c1 {
                  %28 = arith.ceildivui %25, %c8 : index
                  scf.for %arg9 = %c0 to %28 step %c1 {
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                    %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                    %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 8)>(%arg5, %arg6, %arg9)
                    %32 = arith.muli %29, %20 : index
                    %33 = arith.muli %31, %22 : index
                    %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                    %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %41 = arith.muli %30, %21 : index
                    %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %45 : memref<?x?xf16>
                    %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %43 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %56 : memref<?x?xf16>
                    %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %66 = arith.muli %arg3, %c4 : index
                    %67 = arith.muli %arg4, %c2 : index
                    %68 = arith.addi %arg6, %67 : index
                    %69 = arith.addi %66, %c3 : index
                    %70 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%66, %68], LR : [%69, %68]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %71 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %71 {
                      %72 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %73 = loom.sync ins(%70 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %64 : memref<?x?x?xf16>
                      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%73 : tensor<?x?x?xf16>) outs(%72 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %78 = arith.addf %in, %out : f16
                        linalg.yield %78 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %62 : memref<?x?x?xf16>
                      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %52 : memref<?x?xf16>
                      %76 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %50 : memref<?x?xf16>
                    }
                  } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y2y4__d0i2_d1i2_d2i1_d3i0__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c3 = arith.constant 3 : index
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
          affine.parallel (%arg5) = (0) to (4) {
            affine.parallel (%arg6) = (0) to (2) {
              %26 = arith.ceildivui %23, %c4 : index
              scf.for %arg7 = %c0 to %26 step %c1 {
                %27 = arith.ceildivui %24, %c2 : index
                scf.for %arg8 = %c0 to %27 step %c1 {
                  %28 = arith.ceildivui %25, %c8 : index
                  scf.for %arg9 = %c0 to %28 step %c1 {
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                    %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                    %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 8)>(%arg5, %arg6, %arg9)
                    %32 = arith.muli %29, %20 : index
                    %33 = arith.muli %31, %22 : index
                    %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                    %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %41 = arith.muli %30, %21 : index
                    %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %45 : memref<?x?xf16>
                    %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %43 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %56 : memref<?x?xf16>
                    %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %66 = arith.muli %arg4, %c4 : index
                    %67 = arith.muli %arg3, %c2 : index
                    %68 = arith.addi %arg6, %67 : index
                    %69 = arith.addi %66, %c3 : index
                    %70 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%66, %68], LR : [%69, %68]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %71 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %71 {
                      %72 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %73 = loom.sync ins(%70 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %64 : memref<?x?x?xf16>
                      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%73 : tensor<?x?x?xf16>) outs(%72 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %78 = arith.addf %in, %out : f16
                        linalg.yield %78 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %62 : memref<?x?x?xf16>
                      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %52 : memref<?x?xf16>
                      %76 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %50 : memref<?x?xf16>
                    }
                  } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y4y2__d0i2_d1i2_d2i0_d3i1__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
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
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.parallel (%arg6) = (0) to (4) {
              %26 = arith.ceildivui %23, %c2 : index
              scf.for %arg7 = %c0 to %26 step %c1 {
                %27 = arith.ceildivui %24, %c2 : index
                scf.for %arg8 = %c0 to %27 step %c1 {
                  %28 = arith.ceildivui %25, %c16 : index
                  scf.for %arg9 = %c0 to %28 step %c1 {
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                    %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                    %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 16)>(%arg5, %arg6, %arg9)
                    %32 = arith.muli %29, %20 : index
                    %33 = arith.muli %31, %22 : index
                    %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                    %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %41 = arith.muli %30, %21 : index
                    %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %45 : memref<?x?xf16>
                    %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %43 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %56 : memref<?x?xf16>
                    %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %66 = arith.muli %arg3, %c4 : index
                    %67 = arith.muli %arg4, %c4 : index
                    %68 = arith.addi %arg6, %67 : index
                    %69 = arith.addi %66, %c3 : index
                    %70 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%66, %68], LR : [%69, %68]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %71 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %71 {
                      %72 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %73 = loom.sync ins(%70 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %64 : memref<?x?x?xf16>
                      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%73 : tensor<?x?x?xf16>) outs(%72 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %78 = arith.addf %in, %out : f16
                        linalg.yield %78 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %62 : memref<?x?x?xf16>
                      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %52 : memref<?x?xf16>
                      %76 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %50 : memref<?x?xf16>
                    }
                  } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y4y2__d0i2_d1i2_d2i1_d3i0__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
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
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.parallel (%arg6) = (0) to (4) {
              %26 = arith.ceildivui %23, %c2 : index
              scf.for %arg7 = %c0 to %26 step %c1 {
                %27 = arith.ceildivui %24, %c2 : index
                scf.for %arg8 = %c0 to %27 step %c1 {
                  %28 = arith.ceildivui %25, %c16 : index
                  scf.for %arg9 = %c0 to %28 step %c1 {
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                    %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                    %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 16)>(%arg5, %arg6, %arg9)
                    %32 = arith.muli %29, %20 : index
                    %33 = arith.muli %31, %22 : index
                    %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                    %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %41 = arith.muli %30, %21 : index
                    %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %45 : memref<?x?xf16>
                    %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %43 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %56 : memref<?x?xf16>
                    %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %66 = arith.muli %arg4, %c4 : index
                    %67 = arith.muli %arg3, %c4 : index
                    %68 = arith.addi %arg6, %67 : index
                    %69 = arith.addi %66, %c3 : index
                    %70 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%66, %68], LR : [%69, %68]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %71 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %71 {
                      %72 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %73 = loom.sync ins(%70 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %64 : memref<?x?x?xf16>
                      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%73 : tensor<?x?x?xf16>) outs(%72 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %78 = arith.addf %in, %out : f16
                        linalg.yield %78 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %62 : memref<?x?x?xf16>
                      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %52 : memref<?x?xf16>
                      %76 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %50 : memref<?x?xf16>
                    }
                  } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x2x2_y8__d0i2_d1i2_d2i0_d3i1__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
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
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.parallel (%arg6) = (0) to (8) {
              %26 = arith.ceildivui %23, %c2 : index
              scf.for %arg7 = %c0 to %26 step %c1 {
                %27 = arith.ceildivui %24, %c2 : index
                scf.for %arg8 = %c0 to %27 step %c1 {
                  %28 = arith.ceildivui %25, %c16 : index
                  scf.for %arg9 = %c0 to %28 step %c1 {
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                    %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                    %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg5, %arg6, %arg9)
                    %32 = arith.muli %29, %20 : index
                    %33 = arith.muli %31, %22 : index
                    %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                    %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %41 = arith.muli %30, %21 : index
                    %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %45 : memref<?x?xf16>
                    %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %43 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %56 : memref<?x?xf16>
                    %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %66 = arith.muli %arg3, %c2 : index
                    %67 = arith.muli %arg4, %c4 : index
                    %68 = arith.addi %66, %67 : index
                    %69 = arith.addi %66, %c1 : index
                    %70 = arith.addi %69, %67 : index
                    %71 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%68, %arg6], LR : [%70, %arg6]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %72 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %72 {
                      %73 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %74 = loom.sync ins(%71 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %64 : memref<?x?x?xf16>
                      %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%74 : tensor<?x?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %79 = arith.addf %in, %out : f16
                        linalg.yield %79 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %62 : memref<?x?x?xf16>
                      %76 = loom.sync ins(%75 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %52 : memref<?x?xf16>
                      %77 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %78 = loom.bufferize_to_memref %76 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %50 : memref<?x?xf16>
                    }
                  } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x2x2_y8__d0i2_d1i2_d2i1_d3i0__f012(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
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
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.parallel (%arg6) = (0) to (8) {
              %26 = arith.ceildivui %23, %c2 : index
              scf.for %arg7 = %c0 to %26 step %c1 {
                %27 = arith.ceildivui %24, %c2 : index
                scf.for %arg8 = %c0 to %27 step %c1 {
                  %28 = arith.ceildivui %25, %c16 : index
                  scf.for %arg9 = %c0 to %28 step %c1 {
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                    %30 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                    %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg5, %arg6, %arg9)
                    %32 = arith.muli %29, %20 : index
                    %33 = arith.muli %31, %22 : index
                    %34 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                    %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %39 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %40 = loom.sync ins(%39 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %41 = arith.muli %30, %21 : index
                    %42 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %43 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %44 = loom.init_tensor %43[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %45 = loom.semaphore_take %42 : memref<?x?xf16> -> memref<?x?xf16>
                    %46 = loom.subview %arg1[%33, %41] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %46, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %47 = loom.bufferize_to_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %48 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %45 : memref<?x?xf16>
                    %49 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %51 = loom.init_tensor %50[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %54 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %59 = linalg.matmul ins(%40, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %43 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %60 = loom.sync ins(%59 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %56 : memref<?x?xf16>
                    %61 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %62 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %63 = loom.init_tensor %62[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %64 = loom.semaphore_take %61 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %66 = arith.muli %arg4, %c2 : index
                    %67 = arith.muli %arg3, %c4 : index
                    %68 = arith.addi %66, %67 : index
                    %69 = arith.addi %66, %c1 : index
                    %70 = arith.addi %69, %67 : index
                    %71 = loom.gather ins(%60 : tensor<?x?xf16>) outs(%65 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%68, %arg6], LR : [%70, %arg6]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %72 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %72 {
                      %73 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %74 = loom.sync ins(%71 : tensor<?x?x?xf16>) outs(%63 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %64 : memref<?x?x?xf16>
                      %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%74 : tensor<?x?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %79 = arith.addf %in, %out : f16
                        linalg.yield %79 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %62 : memref<?x?x?xf16>
                      %76 = loom.sync ins(%75 : tensor<?x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %52 : memref<?x?xf16>
                      %77 = loom.subview %arg2[%32, %41] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %78 = loom.bufferize_to_memref %76 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %50 : memref<?x?xf16>
                    }
                  } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
