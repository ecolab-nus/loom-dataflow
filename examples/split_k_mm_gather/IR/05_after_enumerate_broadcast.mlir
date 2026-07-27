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
    func.func @split_k_matmul_gather__x8_y2y4__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc2_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %39 = arith.muli %arg4, %c2 : index
                  %40 = arith.addi %arg3, %39 : index
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %40], LR : [%arg5, %40]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %41 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %42 = loom.sync ins(%41 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %43 = arith.muli %30, %21 : index
                  %44 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %45 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.init_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %48 = loom.subview %arg1[%33, %43] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  %49 = arith.addi %39, %c1 : index
                  loom.copy %48, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %39], LR : [%arg5, %49]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %50 = loom.bufferize_to_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %51 = loom.sync ins(%50 : tensor<?x?xf16>) outs(%46 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %47 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %56 = loom.init_tensor %55[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %57 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %58 = loom.init_tensor %57[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %59 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %60 = loom.init_tensor %59[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %61 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %62 = linalg.matmul ins(%42, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %63 = loom.sync ins(%62 : tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %59 : memref<?x?xf16>
                  %64 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %66 = loom.init_tensor %65[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %67 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %68 = loom.init_tensor %67[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %69 = loom.gather ins(%63 : tensor<?x?xf16>) outs(%68 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %40], LR : [%c7, %40]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %57 : memref<?x?xf16>
                  %70 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %70 {
                    %71 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %72 = loom.sync ins(%69 : tensor<?x?x?xf16>) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %67 : memref<?x?x?xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%72 : tensor<?x?x?xf16>) outs(%71 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %77 = arith.addf %in, %out : f16
                      linalg.yield %77 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %65 : memref<?x?x?xf16>
                    %74 = loom.sync ins(%73 : tensor<?x?xf16>) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %55 : memref<?x?xf16>
                    %75 = loom.subview %arg2[%32, %43] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %76 = loom.bufferize_to_memref %74 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %76, %75 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %40], LR : [%arg5, %40]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y2y4__d0i2_d1i1_d2i0__f012__dim_y_level0_bc2_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %39 = arith.muli %arg3, %c2 : index
                  %40 = arith.addi %39, %c1 : index
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %39], LR : [%arg5, %40]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %41 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %42 = loom.sync ins(%41 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %43 = arith.muli %30, %21 : index
                  %44 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %45 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.init_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %48 = loom.subview %arg1[%33, %43] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  %49 = arith.addi %arg4, %39 : index
                  loom.copy %48, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %49], LR : [%arg5, %49]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %50 = loom.bufferize_to_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %51 = loom.sync ins(%50 : tensor<?x?xf16>) outs(%46 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %47 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %56 = loom.init_tensor %55[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %57 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %58 = loom.init_tensor %57[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %59 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %60 = loom.init_tensor %59[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %61 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %62 = linalg.matmul ins(%42, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %63 = loom.sync ins(%62 : tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %59 : memref<?x?xf16>
                  %64 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %66 = loom.init_tensor %65[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %67 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %68 = loom.init_tensor %67[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %69 = loom.gather ins(%63 : tensor<?x?xf16>) outs(%68 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %49], LR : [%c7, %49]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %57 : memref<?x?xf16>
                  %70 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %70 {
                    %71 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %72 = loom.sync ins(%69 : tensor<?x?x?xf16>) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %67 : memref<?x?x?xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%72 : tensor<?x?x?xf16>) outs(%71 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %77 = arith.addf %in, %out : f16
                      linalg.yield %77 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %65 : memref<?x?x?xf16>
                    %74 = loom.sync ins(%73 : tensor<?x?xf16>) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %55 : memref<?x?xf16>
                    %75 = loom.subview %arg2[%32, %43] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %76 = loom.bufferize_to_memref %74 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %76, %75 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %49], LR : [%arg5, %49]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc4_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %39 = arith.muli %arg4, %c4 : index
                  %40 = arith.addi %arg3, %39 : index
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %40], LR : [%arg5, %40]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %41 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %42 = loom.sync ins(%41 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %43 = arith.muli %30, %21 : index
                  %44 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %45 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.init_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %48 = loom.subview %arg1[%33, %43] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  %49 = arith.addi %39, %c3 : index
                  loom.copy %48, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %39], LR : [%arg5, %49]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %50 = loom.bufferize_to_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %51 = loom.sync ins(%50 : tensor<?x?xf16>) outs(%46 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %47 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %56 = loom.init_tensor %55[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %57 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %58 = loom.init_tensor %57[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %59 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %60 = loom.init_tensor %59[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %61 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %62 = linalg.matmul ins(%42, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %63 = loom.sync ins(%62 : tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %59 : memref<?x?xf16>
                  %64 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %66 = loom.init_tensor %65[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %67 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %68 = loom.init_tensor %67[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %69 = loom.gather ins(%63 : tensor<?x?xf16>) outs(%68 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %40], LR : [%c7, %40]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %57 : memref<?x?xf16>
                  %70 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %70 {
                    %71 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %72 = loom.sync ins(%69 : tensor<?x?x?xf16>) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %67 : memref<?x?x?xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%72 : tensor<?x?x?xf16>) outs(%71 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %77 = arith.addf %in, %out : f16
                      linalg.yield %77 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %65 : memref<?x?x?xf16>
                    %74 = loom.sync ins(%73 : tensor<?x?xf16>) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %55 : memref<?x?xf16>
                    %75 = loom.subview %arg2[%32, %43] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %76 = loom.bufferize_to_memref %74 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %76, %75 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %40], LR : [%arg5, %40]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i1_d2i0__f012__dim_y_level0_bc4_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %39 = arith.muli %arg3, %c4 : index
                  %40 = arith.addi %39, %c3 : index
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %39], LR : [%arg5, %40]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %41 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %42 = loom.sync ins(%41 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %43 = arith.muli %30, %21 : index
                  %44 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %45 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.init_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %48 = loom.subview %arg1[%33, %43] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  %49 = arith.addi %arg4, %39 : index
                  loom.copy %48, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %49], LR : [%arg5, %49]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %50 = loom.bufferize_to_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %51 = loom.sync ins(%50 : tensor<?x?xf16>) outs(%46 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %47 : memref<?x?xf16>
                  %52 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %54 = loom.init_tensor %53[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %56 = loom.init_tensor %55[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %57 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %58 = loom.init_tensor %57[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %59 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
                  %60 = loom.init_tensor %59[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %61 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %62 = linalg.matmul ins(%42, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %63 = loom.sync ins(%62 : tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %59 : memref<?x?xf16>
                  %64 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %66 = loom.init_tensor %65[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %67 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %68 = loom.init_tensor %67[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %69 = loom.gather ins(%63 : tensor<?x?xf16>) outs(%68 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %49], LR : [%c7, %49]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %57 : memref<?x?xf16>
                  %70 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %70 {
                    %71 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %72 = loom.sync ins(%69 : tensor<?x?x?xf16>) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %67 : memref<?x?x?xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%72 : tensor<?x?x?xf16>) outs(%71 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %77 = arith.addf %in, %out : f16
                      linalg.yield %77 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %65 : memref<?x?x?xf16>
                    %74 = loom.sync ins(%73 : tensor<?x?xf16>) outs(%54 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %55 : memref<?x?xf16>
                    %75 = loom.subview %arg2[%32, %43] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %76 = loom.bufferize_to_memref %74 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %76, %75 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %49], LR : [%arg5, %49]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc8_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %39 = arith.muli %arg4, %c2 : index
                  %40 = arith.addi %arg5, %39 : index
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %arg3], LR : [%40, %arg3]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %41 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %42 = loom.sync ins(%41 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %43 = arith.muli %30, %21 : index
                  %44 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %45 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.init_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %48 = loom.subview %arg1[%33, %43] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %48, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%40, %c0], LR : [%40, %c7]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %49 = loom.bufferize_to_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = loom.sync ins(%49 : tensor<?x?xf16>) outs(%46 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %47 : memref<?x?xf16>
                  %51 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %52 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %54 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %56 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %58 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %60 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %61 = linalg.matmul ins(%42, %50 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%60 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %62 = loom.sync ins(%61 : tensor<?x?xf16>) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %58 : memref<?x?xf16>
                  %63 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %64 = loom.semaphore_take %63 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %66 = loom.semaphore_take %63 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %68 = arith.addi %39, %c1 : index
                  %69 = loom.gather ins(%62 : tensor<?x?xf16>) outs(%67 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%39, %arg3], LR : [%68, %arg3]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %56 : memref<?x?xf16>
                  %70 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %70 {
                    %71 = linalg.fill ins(%cst : f16) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %72 = loom.sync ins(%69 : tensor<?x?x?xf16>) outs(%65 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %66 : memref<?x?x?xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%72 : tensor<?x?x?xf16>) outs(%71 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %77 = arith.addf %in, %out : f16
                      linalg.yield %77 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %64 : memref<?x?x?xf16>
                    %74 = loom.sync ins(%73 : tensor<?x?xf16>) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %75 = loom.subview %arg2[%32, %43] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %76 = loom.bufferize_to_memref %74 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %76, %75 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%40, %arg3], LR : [%40, %arg3]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %52 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i1_d2i0__f012__dim_y_level0_bc8_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %39 = arith.muli %arg3, %c2 : index
                  %40 = arith.addi %arg5, %39 : index
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%40, %c0], LR : [%40, %c7]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %41 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %42 = loom.sync ins(%41 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %43 = arith.muli %30, %21 : index
                  %44 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %45 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.init_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %48 = loom.subview %arg1[%33, %43] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %48, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %arg4], LR : [%40, %arg4]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %49 = loom.bufferize_to_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = loom.sync ins(%49 : tensor<?x?xf16>) outs(%46 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %47 : memref<?x?xf16>
                  %51 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %52 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %54 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %56 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %58 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %60 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %61 = linalg.matmul ins(%42, %50 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%60 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %62 = loom.sync ins(%61 : tensor<?x?xf16>) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %58 : memref<?x?xf16>
                  %63 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %64 = loom.semaphore_take %63 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %66 = loom.semaphore_take %63 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %68 = arith.addi %39, %c1 : index
                  %69 = loom.gather ins(%62 : tensor<?x?xf16>) outs(%67 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%39, %arg4], LR : [%68, %arg4]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %56 : memref<?x?xf16>
                  %70 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %70 {
                    %71 = linalg.fill ins(%cst : f16) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %72 = loom.sync ins(%69 : tensor<?x?x?xf16>) outs(%65 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %66 : memref<?x?x?xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%72 : tensor<?x?x?xf16>) outs(%71 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %77 = arith.addf %in, %out : f16
                      linalg.yield %77 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %64 : memref<?x?x?xf16>
                    %74 = loom.sync ins(%73 : tensor<?x?xf16>) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %75 = loom.subview %arg2[%32, %43] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %76 = loom.bufferize_to_memref %74 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %76, %75 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%40, %arg4], LR : [%40, %arg4]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %52 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc8_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %39 = arith.muli %arg4, %c4 : index
                  %40 = arith.addi %arg5, %39 : index
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %arg3], LR : [%40, %arg3]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %41 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %42 = loom.sync ins(%41 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %43 = arith.muli %30, %21 : index
                  %44 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %45 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.init_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %48 = loom.subview %arg1[%33, %43] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %48, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%40, %c0], LR : [%40, %c7]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %49 = loom.bufferize_to_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = loom.sync ins(%49 : tensor<?x?xf16>) outs(%46 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %47 : memref<?x?xf16>
                  %51 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %52 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %54 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %56 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %58 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %60 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %61 = linalg.matmul ins(%42, %50 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%60 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %62 = loom.sync ins(%61 : tensor<?x?xf16>) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %58 : memref<?x?xf16>
                  %63 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %64 = loom.semaphore_take %63 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %66 = loom.semaphore_take %63 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %68 = arith.addi %39, %c3 : index
                  %69 = loom.gather ins(%62 : tensor<?x?xf16>) outs(%67 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%39, %arg3], LR : [%68, %arg3]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %56 : memref<?x?xf16>
                  %70 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %70 {
                    %71 = linalg.fill ins(%cst : f16) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %72 = loom.sync ins(%69 : tensor<?x?x?xf16>) outs(%65 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %66 : memref<?x?x?xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%72 : tensor<?x?x?xf16>) outs(%71 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %77 = arith.addf %in, %out : f16
                      linalg.yield %77 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %64 : memref<?x?x?xf16>
                    %74 = loom.sync ins(%73 : tensor<?x?xf16>) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %75 = loom.subview %arg2[%32, %43] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %76 = loom.bufferize_to_memref %74 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %76, %75 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%40, %arg3], LR : [%40, %arg3]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %52 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i1_d2i0__f012__dim_y_level0_bc8_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                  %36 = loom.init_tensor %35[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %37 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                  %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  %39 = arith.muli %arg3, %c4 : index
                  %40 = arith.addi %arg5, %39 : index
                  loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%40, %c0], LR : [%40, %c7]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %41 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                  %42 = loom.sync ins(%41 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %37 : memref<?x?xf16>
                  %43 = arith.muli %30, %21 : index
                  %44 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                  %45 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %46 = loom.init_tensor %45[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
                  %48 = loom.subview %arg1[%33, %43] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %48, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %arg4], LR : [%40, %arg4]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %49 = loom.bufferize_to_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = loom.sync ins(%49 : tensor<?x?xf16>) outs(%46 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %47 : memref<?x?xf16>
                  %51 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                  %52 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %53 = loom.init_tensor %52[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %54 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %56 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %58 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
                  %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                  %60 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %61 = linalg.matmul ins(%42, %50 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%60 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %45 : memref<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  %62 = loom.sync ins(%61 : tensor<?x?xf16>) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %58 : memref<?x?xf16>
                  %63 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                  %64 = loom.semaphore_take %63 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %65 = loom.init_tensor %64[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %66 = loom.semaphore_take %63 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                  %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                  %68 = arith.addi %39, %c3 : index
                  %69 = loom.gather ins(%62 : tensor<?x?xf16>) outs(%67 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%39, %arg4], LR : [%68, %arg4]) -> tensor<?x?x?xf16>
                  loom.semaphore_give %56 : memref<?x?xf16>
                  %70 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %70 {
                    %71 = linalg.fill ins(%cst : f16) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %72 = loom.sync ins(%69 : tensor<?x?x?xf16>) outs(%65 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    loom.semaphore_give %66 : memref<?x?x?xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%72 : tensor<?x?x?xf16>) outs(%71 : tensor<?x?xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %77 = arith.addf %in, %out : f16
                      linalg.yield %77 : f16
                    } -> tensor<?x?xf16>
                    loom.semaphore_give %64 : memref<?x?x?xf16>
                    %74 = loom.sync ins(%73 : tensor<?x?xf16>) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %54 : memref<?x?xf16>
                    %75 = loom.subview %arg2[%32, %43] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    %76 = loom.bufferize_to_memref %74 : tensor<?x?xf16> -> memref<?x?xf16>
                    loom.copy %76, %75 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%40, %arg4], LR : [%40, %arg4]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %52 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x8_y2y2y2__d0i2_d1i2_d2i0_d3i1__f012__n_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    %39 = arith.muli %arg3, %c2 : index
                    %40 = arith.addi %arg6, %39 : index
                    %41 = arith.muli %arg4, %c4 : index
                    %42 = arith.addi %40, %41 : index
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %42], LR : [%arg5, %42]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %43 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %44 = loom.sync ins(%43 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %45 = arith.muli %30, %21 : index
                    %46 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %48 = loom.init_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %50 = loom.subview %arg1[%33, %45] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %50, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %42], LR : [%arg5, %42]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %51 = loom.bufferize_to_tensor %49[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.sync ins(%51 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %49 : memref<?x?xf16>
                    %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %60 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %62 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %63 = linalg.matmul ins(%44, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %47 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %64 = loom.sync ins(%63 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %60 : memref<?x?xf16>
                    %65 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %68 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %69 = loom.init_tensor %68[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %70 = loom.gather ins(%64 : tensor<?x?xf16>) outs(%69 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %42], LR : [%c7, %42]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %58 : memref<?x?xf16>
                    %71 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %71 {
                      %72 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %73 = loom.sync ins(%70 : tensor<?x?x?xf16>) outs(%67 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %68 : memref<?x?x?xf16>
                      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%73 : tensor<?x?x?xf16>) outs(%72 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %78 = arith.addf %in, %out : f16
                        linalg.yield %78 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %66 : memref<?x?x?xf16>
                      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %56 : memref<?x?xf16>
                      %76 = loom.subview %arg2[%32, %45] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %42], LR : [%arg5, %42]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %54 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x8_y2y2y2__d0i2_d1i2_d2i1_d3i0__f012__n_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    %39 = arith.muli %arg4, %c2 : index
                    %40 = arith.addi %arg6, %39 : index
                    %41 = arith.muli %arg3, %c4 : index
                    %42 = arith.addi %40, %41 : index
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %42], LR : [%arg5, %42]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %43 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %44 = loom.sync ins(%43 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %45 = arith.muli %30, %21 : index
                    %46 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %48 = loom.init_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %50 = loom.subview %arg1[%33, %45] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %50, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %42], LR : [%arg5, %42]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %51 = loom.bufferize_to_tensor %49[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.sync ins(%51 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %49 : memref<?x?xf16>
                    %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %60 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %62 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %63 = linalg.matmul ins(%44, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %47 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %64 = loom.sync ins(%63 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %60 : memref<?x?xf16>
                    %65 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %68 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %69 = loom.init_tensor %68[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %70 = loom.gather ins(%64 : tensor<?x?xf16>) outs(%69 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%c0, %42], LR : [%c7, %42]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %58 : memref<?x?xf16>
                    %71 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %71 {
                      %72 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %73 = loom.sync ins(%70 : tensor<?x?x?xf16>) outs(%67 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %68 : memref<?x?x?xf16>
                      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%73 : tensor<?x?x?xf16>) outs(%72 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %78 = arith.addf %in, %out : f16
                        linalg.yield %78 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %66 : memref<?x?x?xf16>
                      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %56 : memref<?x?xf16>
                      %76 = loom.subview %arg2[%32, %45] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %42], LR : [%arg5, %42]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %54 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x2x4_y2y4__d0i2_d1i2_d2i0_d3i1__f012__n_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    %39 = arith.muli %arg3, %c2 : index
                    %40 = arith.addi %arg5, %39 : index
                    %41 = arith.muli %arg4, %c2 : index
                    %42 = arith.addi %arg6, %41 : index
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %43 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %44 = loom.sync ins(%43 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %45 = arith.muli %30, %21 : index
                    %46 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %48 = loom.init_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %50 = loom.subview %arg1[%33, %45] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %50, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %51 = loom.bufferize_to_tensor %49[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.sync ins(%51 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %49 : memref<?x?xf16>
                    %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %60 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %62 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %63 = linalg.matmul ins(%44, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %47 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %64 = loom.sync ins(%63 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %60 : memref<?x?xf16>
                    %65 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %68 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %69 = loom.init_tensor %68[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %70 = arith.addi %39, %c1 : index
                    %71 = loom.gather ins(%64 : tensor<?x?xf16>) outs(%69 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%39, %42], LR : [%70, %42]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %58 : memref<?x?xf16>
                    %72 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %72 {
                      %73 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %74 = loom.sync ins(%71 : tensor<?x?x?xf16>) outs(%67 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %68 : memref<?x?x?xf16>
                      %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%74 : tensor<?x?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %79 = arith.addf %in, %out : f16
                        linalg.yield %79 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %66 : memref<?x?x?xf16>
                      %76 = loom.sync ins(%75 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %56 : memref<?x?xf16>
                      %77 = loom.subview %arg2[%32, %45] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %78 = loom.bufferize_to_memref %76 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %54 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x2x4_y2y4__d0i2_d1i2_d2i1_d3i0__f012__n_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    %39 = arith.muli %arg4, %c2 : index
                    %40 = arith.addi %arg5, %39 : index
                    %41 = arith.muli %arg3, %c2 : index
                    %42 = arith.addi %arg6, %41 : index
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %43 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %44 = loom.sync ins(%43 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %45 = arith.muli %30, %21 : index
                    %46 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %48 = loom.init_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %50 = loom.subview %arg1[%33, %45] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %50, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %51 = loom.bufferize_to_tensor %49[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.sync ins(%51 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %49 : memref<?x?xf16>
                    %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %60 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %62 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %63 = linalg.matmul ins(%44, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %47 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %64 = loom.sync ins(%63 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %60 : memref<?x?xf16>
                    %65 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %68 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %69 = loom.init_tensor %68[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %70 = arith.addi %39, %c1 : index
                    %71 = loom.gather ins(%64 : tensor<?x?xf16>) outs(%69 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%39, %42], LR : [%70, %42]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %58 : memref<?x?xf16>
                    %72 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %72 {
                      %73 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %74 = loom.sync ins(%71 : tensor<?x?x?xf16>) outs(%67 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %68 : memref<?x?x?xf16>
                      %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%74 : tensor<?x?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %79 = arith.addf %in, %out : f16
                        linalg.yield %79 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %66 : memref<?x?x?xf16>
                      %76 = loom.sync ins(%75 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %56 : memref<?x?xf16>
                      %77 = loom.subview %arg2[%32, %45] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %78 = loom.bufferize_to_memref %76 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %54 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x2x4_y4y2__d0i2_d1i2_d2i0_d3i1__f012__n_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    %39 = arith.muli %arg3, %c2 : index
                    %40 = arith.addi %arg5, %39 : index
                    %41 = arith.muli %arg4, %c4 : index
                    %42 = arith.addi %arg6, %41 : index
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %43 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %44 = loom.sync ins(%43 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %45 = arith.muli %30, %21 : index
                    %46 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %48 = loom.init_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %50 = loom.subview %arg1[%33, %45] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %50, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %51 = loom.bufferize_to_tensor %49[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.sync ins(%51 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %49 : memref<?x?xf16>
                    %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %60 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %62 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %63 = linalg.matmul ins(%44, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %47 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %64 = loom.sync ins(%63 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %60 : memref<?x?xf16>
                    %65 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %68 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %69 = loom.init_tensor %68[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %70 = arith.addi %39, %c1 : index
                    %71 = loom.gather ins(%64 : tensor<?x?xf16>) outs(%69 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%39, %42], LR : [%70, %42]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %58 : memref<?x?xf16>
                    %72 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %72 {
                      %73 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %74 = loom.sync ins(%71 : tensor<?x?x?xf16>) outs(%67 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %68 : memref<?x?x?xf16>
                      %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%74 : tensor<?x?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %79 = arith.addf %in, %out : f16
                        linalg.yield %79 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %66 : memref<?x?x?xf16>
                      %76 = loom.sync ins(%75 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %56 : memref<?x?xf16>
                      %77 = loom.subview %arg2[%32, %45] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %78 = loom.bufferize_to_memref %76 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %54 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x2x4_y4y2__d0i2_d1i2_d2i1_d3i0__f012__n_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    %39 = arith.muli %arg4, %c2 : index
                    %40 = arith.addi %arg5, %39 : index
                    %41 = arith.muli %arg3, %c4 : index
                    %42 = arith.addi %arg6, %41 : index
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %43 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %44 = loom.sync ins(%43 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %45 = arith.muli %30, %21 : index
                    %46 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %48 = loom.init_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %50 = loom.subview %arg1[%33, %45] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %50, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %51 = loom.bufferize_to_tensor %49[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.sync ins(%51 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %49 : memref<?x?xf16>
                    %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %60 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %62 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %63 = linalg.matmul ins(%44, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %47 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %64 = loom.sync ins(%63 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %60 : memref<?x?xf16>
                    %65 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %68 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %69 = loom.init_tensor %68[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %70 = arith.addi %39, %c1 : index
                    %71 = loom.gather ins(%64 : tensor<?x?xf16>) outs(%69 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%39, %42], LR : [%70, %42]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %58 : memref<?x?xf16>
                    %72 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %72 {
                      %73 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %74 = loom.sync ins(%71 : tensor<?x?x?xf16>) outs(%67 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %68 : memref<?x?x?xf16>
                      %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%74 : tensor<?x?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %79 = arith.addf %in, %out : f16
                        linalg.yield %79 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %66 : memref<?x?x?xf16>
                      %76 = loom.sync ins(%75 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %56 : memref<?x?xf16>
                      %77 = loom.subview %arg2[%32, %45] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %78 = loom.bufferize_to_memref %76 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %54 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x4x2_y2y4__d0i2_d1i2_d2i0_d3i1__f012__n_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    %39 = arith.muli %arg3, %c4 : index
                    %40 = arith.addi %arg5, %39 : index
                    %41 = arith.muli %arg4, %c2 : index
                    %42 = arith.addi %arg6, %41 : index
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %43 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %44 = loom.sync ins(%43 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %45 = arith.muli %30, %21 : index
                    %46 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %48 = loom.init_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %50 = loom.subview %arg1[%33, %45] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %50, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %51 = loom.bufferize_to_tensor %49[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.sync ins(%51 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %49 : memref<?x?xf16>
                    %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %60 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %62 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %63 = linalg.matmul ins(%44, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %47 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %64 = loom.sync ins(%63 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %60 : memref<?x?xf16>
                    %65 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %68 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %69 = loom.init_tensor %68[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %70 = arith.addi %39, %c3 : index
                    %71 = loom.gather ins(%64 : tensor<?x?xf16>) outs(%69 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%39, %42], LR : [%70, %42]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %58 : memref<?x?xf16>
                    %72 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %72 {
                      %73 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %74 = loom.sync ins(%71 : tensor<?x?x?xf16>) outs(%67 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %68 : memref<?x?x?xf16>
                      %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%74 : tensor<?x?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %79 = arith.addf %in, %out : f16
                        linalg.yield %79 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %66 : memref<?x?x?xf16>
                      %76 = loom.sync ins(%75 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %56 : memref<?x?xf16>
                      %77 = loom.subview %arg2[%32, %45] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %78 = loom.bufferize_to_memref %76 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %54 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x4x2_y2y4__d0i2_d1i2_d2i1_d3i0__f012__n_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    %39 = arith.muli %arg4, %c4 : index
                    %40 = arith.addi %arg5, %39 : index
                    %41 = arith.muli %arg3, %c2 : index
                    %42 = arith.addi %arg6, %41 : index
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %43 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %44 = loom.sync ins(%43 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %45 = arith.muli %30, %21 : index
                    %46 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %48 = loom.init_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %50 = loom.subview %arg1[%33, %45] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %50, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %51 = loom.bufferize_to_tensor %49[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.sync ins(%51 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %49 : memref<?x?xf16>
                    %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %60 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %62 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %63 = linalg.matmul ins(%44, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %47 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %64 = loom.sync ins(%63 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %60 : memref<?x?xf16>
                    %65 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %68 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %69 = loom.init_tensor %68[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %70 = arith.addi %39, %c3 : index
                    %71 = loom.gather ins(%64 : tensor<?x?xf16>) outs(%69 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%39, %42], LR : [%70, %42]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %58 : memref<?x?xf16>
                    %72 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %72 {
                      %73 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %74 = loom.sync ins(%71 : tensor<?x?x?xf16>) outs(%67 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %68 : memref<?x?x?xf16>
                      %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%74 : tensor<?x?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %79 = arith.addf %in, %out : f16
                        linalg.yield %79 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %66 : memref<?x?x?xf16>
                      %76 = loom.sync ins(%75 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %56 : memref<?x?xf16>
                      %77 = loom.subview %arg2[%32, %45] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %78 = loom.bufferize_to_memref %76 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %54 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x4x2_y4y2__d0i2_d1i2_d2i0_d3i1__f012__n_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    %39 = arith.muli %arg3, %c4 : index
                    %40 = arith.addi %arg5, %39 : index
                    %41 = arith.muli %arg4, %c4 : index
                    %42 = arith.addi %arg6, %41 : index
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %43 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %44 = loom.sync ins(%43 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %45 = arith.muli %30, %21 : index
                    %46 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %48 = loom.init_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %50 = loom.subview %arg1[%33, %45] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %50, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %51 = loom.bufferize_to_tensor %49[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.sync ins(%51 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %49 : memref<?x?xf16>
                    %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %60 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %62 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %63 = linalg.matmul ins(%44, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %47 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %64 = loom.sync ins(%63 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %60 : memref<?x?xf16>
                    %65 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %68 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %69 = loom.init_tensor %68[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %70 = arith.addi %39, %c3 : index
                    %71 = loom.gather ins(%64 : tensor<?x?xf16>) outs(%69 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%39, %42], LR : [%70, %42]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %58 : memref<?x?xf16>
                    %72 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %72 {
                      %73 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %74 = loom.sync ins(%71 : tensor<?x?x?xf16>) outs(%67 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %68 : memref<?x?x?xf16>
                      %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%74 : tensor<?x?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %79 = arith.addf %in, %out : f16
                        linalg.yield %79 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %66 : memref<?x?x?xf16>
                      %76 = loom.sync ins(%75 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %56 : memref<?x?xf16>
                      %77 = loom.subview %arg2[%32, %45] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %78 = loom.bufferize_to_memref %76 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %54 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x4x2_y4y2__d0i2_d1i2_d2i1_d3i0__f012__n_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    %39 = arith.muli %arg4, %c4 : index
                    %40 = arith.addi %arg5, %39 : index
                    %41 = arith.muli %arg3, %c4 : index
                    %42 = arith.addi %arg6, %41 : index
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %43 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %44 = loom.sync ins(%43 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %45 = arith.muli %30, %21 : index
                    %46 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %48 = loom.init_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %50 = loom.subview %arg1[%33, %45] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %50, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %51 = loom.bufferize_to_tensor %49[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.sync ins(%51 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %49 : memref<?x?xf16>
                    %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %60 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %62 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %63 = linalg.matmul ins(%44, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %47 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %64 = loom.sync ins(%63 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %60 : memref<?x?xf16>
                    %65 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %68 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %69 = loom.init_tensor %68[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %70 = arith.addi %39, %c3 : index
                    %71 = loom.gather ins(%64 : tensor<?x?xf16>) outs(%69 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%39, %42], LR : [%70, %42]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %58 : memref<?x?xf16>
                    %72 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %72 {
                      %73 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %74 = loom.sync ins(%71 : tensor<?x?x?xf16>) outs(%67 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %68 : memref<?x?x?xf16>
                      %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%74 : tensor<?x?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %79 = arith.addf %in, %out : f16
                        linalg.yield %79 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %66 : memref<?x?x?xf16>
                      %76 = loom.sync ins(%75 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %56 : memref<?x?xf16>
                      %77 = loom.subview %arg2[%32, %45] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %78 = loom.bufferize_to_memref %76 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%40, %42], LR : [%40, %42]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %54 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x2x2x2_y8__d0i2_d1i2_d2i0_d3i1__f012__n_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    %39 = arith.muli %arg3, %c2 : index
                    %40 = arith.addi %arg5, %39 : index
                    %41 = arith.muli %arg4, %c4 : index
                    %42 = arith.addi %40, %41 : index
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%42, %arg6], LR : [%42, %arg6]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %43 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %44 = loom.sync ins(%43 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %45 = arith.muli %30, %21 : index
                    %46 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %48 = loom.init_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %50 = loom.subview %arg1[%33, %45] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %50, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%42, %arg6], LR : [%42, %arg6]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %51 = loom.bufferize_to_tensor %49[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.sync ins(%51 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %49 : memref<?x?xf16>
                    %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %60 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %62 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %63 = linalg.matmul ins(%44, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %47 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %64 = loom.sync ins(%63 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %60 : memref<?x?xf16>
                    %65 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %68 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %69 = loom.init_tensor %68[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %70 = arith.addi %39, %41 : index
                    %71 = arith.addi %39, %c1 : index
                    %72 = arith.addi %71, %41 : index
                    %73 = loom.gather ins(%64 : tensor<?x?xf16>) outs(%69 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%70, %arg6], LR : [%72, %arg6]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %58 : memref<?x?xf16>
                    %74 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %74 {
                      %75 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %76 = loom.sync ins(%73 : tensor<?x?x?xf16>) outs(%67 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %68 : memref<?x?x?xf16>
                      %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%76 : tensor<?x?x?xf16>) outs(%75 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %81 = arith.addf %in, %out : f16
                        linalg.yield %81 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %66 : memref<?x?x?xf16>
                      %78 = loom.sync ins(%77 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %56 : memref<?x?xf16>
                      %79 = loom.subview %arg2[%32, %45] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %80 = loom.bufferize_to_memref %78 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %80, %79 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%42, %arg6], LR : [%42, %arg6]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %54 : memref<?x?xf16>
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
    func.func @split_k_matmul_gather__x2x2x2_y8__d0i2_d1i2_d2i1_d3i0__f012__n_n_n(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
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
                    %38 = loom.subview %arg0[%32, %33] [%20, %22] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                    %39 = arith.muli %arg4, %c2 : index
                    %40 = arith.addi %arg5, %39 : index
                    %41 = arith.muli %arg3, %c4 : index
                    %42 = arith.addi %40, %41 : index
                    loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%42, %arg6], LR : [%42, %arg6]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                    %43 = loom.bufferize_to_tensor %37[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                    %44 = loom.sync ins(%43 : tensor<?x?xf16>) outs(%36 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %37 : memref<?x?xf16>
                    %45 = arith.muli %30, %21 : index
                    %46 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                    %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %48 = loom.init_tensor %47[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
                    %50 = loom.subview %arg1[%33, %45] [%22, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                    loom.copy %50, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%42, %arg6], LR : [%42, %arg6]) : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                    %51 = loom.bufferize_to_tensor %49[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %52 = loom.sync ins(%51 : tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %49 : memref<?x?xf16>
                    %53 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                    %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %55 = loom.init_tensor %54[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %56 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %57 = loom.init_tensor %56[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %58 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %59 = loom.init_tensor %58[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %60 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
                    %61 = loom.init_tensor %60[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                    %62 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    %63 = linalg.matmul ins(%44, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %47 : memref<?x?xf16>
                    loom.semaphore_give %35 : memref<?x?xf16>
                    %64 = loom.sync ins(%63 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
                    loom.semaphore_give %60 : memref<?x?xf16>
                    %65 = loom.alloc [%25, %20, %21] on @L1 : memref<?x?x?xf16>
                    %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %67 = loom.init_tensor %66[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %68 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                    %69 = loom.init_tensor %68[%25, %20, %21] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                    %70 = arith.addi %39, %41 : index
                    %71 = arith.addi %39, %c1 : index
                    %72 = arith.addi %71, %41 : index
                    %73 = loom.gather ins(%64 : tensor<?x?xf16>) outs(%69 : tensor<?x?x?xf16>) across(%arg5 : index) region : (UL : [%70, %arg6], LR : [%72, %arg6]) -> tensor<?x?x?xf16>
                    loom.semaphore_give %58 : memref<?x?xf16>
                    %74 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %74 {
                      %75 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      %76 = loom.sync ins(%73 : tensor<?x?x?xf16>) outs(%67 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                      loom.semaphore_give %68 : memref<?x?x?xf16>
                      %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%76 : tensor<?x?x?xf16>) outs(%75 : tensor<?x?xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %81 = arith.addf %in, %out : f16
                        linalg.yield %81 : f16
                      } -> tensor<?x?xf16>
                      loom.semaphore_give %66 : memref<?x?x?xf16>
                      %78 = loom.sync ins(%77 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                      loom.semaphore_give %56 : memref<?x?xf16>
                      %79 = loom.subview %arg2[%32, %45] [%20, %21] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      %80 = loom.bufferize_to_memref %78 : tensor<?x?xf16> -> memref<?x?xf16>
                      loom.copy %80, %79 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%42, %arg6], LR : [%42, %arg6]) : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %54 : memref<?x?xf16>
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
