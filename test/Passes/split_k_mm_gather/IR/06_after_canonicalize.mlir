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
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y2y4__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc2_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %22 = arith.muli %20, %c64 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.init_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %27 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %29 = arith.muli %arg4, %c2 : index
                %30 = arith.addi %arg3, %29 : index
                loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %30], LR : [%arg5, %30]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %31 = loom.bufferize_to_tensor %27[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %32 = loom.sync ins(%31 : tensor<64x512xf16>) outs(%26 : tensor<64x512xf16>) -> tensor<64x512xf16>
                loom.semaphore_give %27 : memref<64x512xf16>
                %33 = arith.muli %21, %c32 : index
                %34 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %35 = loom.semaphore_take %34 : memref<512x32xf16> -> memref<512x32xf16>
                %36 = loom.init_tensor %35[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.semaphore_take %34 : memref<512x32xf16> -> memref<512x32xf16>
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                %39 = arith.addi %29, %c1 : index
                loom.copy %reinterpret_cast_0, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %29], LR : [%arg5, %39]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                %40 = loom.bufferize_to_tensor %37[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %41 = loom.sync ins(%40 : tensor<512x32xf16>) outs(%36 : tensor<512x32xf16>) -> tensor<512x32xf16>
                loom.semaphore_give %37 : memref<512x32xf16>
                %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %44 = loom.init_tensor %43[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %47 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %49 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %51 = linalg.fill ins(%cst : f16) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                %52 = linalg.matmul ins(%32, %41 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%51 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %35 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %53 = loom.sync ins(%52 : tensor<64x32xf16>) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %49 : memref<64x32xf16>
                %54 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                %55 = loom.semaphore_take %54 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                %56 = loom.init_tensor %55[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                %57 = loom.semaphore_take %54 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                %59 = loom.gather ins(%53 : tensor<64x32xf16>) outs(%58 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%c0, %30], LR : [%c7, %30]) -> tensor<8x64x32xf16>
                loom.semaphore_give %47 : memref<64x32xf16>
                %60 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %60 {
                  %61 = linalg.fill ins(%cst : f16) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %62 = loom.sync ins(%59 : tensor<8x64x32xf16>) outs(%56 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                  loom.semaphore_give %57 : memref<8x64x32xf16>
                  %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%62 : tensor<8x64x32xf16>) outs(%61 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %67 = arith.addf %in, %out : f16
                    linalg.yield %67 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %55 : memref<8x64x32xf16>
                  %64 = loom.sync ins(%63 : tensor<64x32xf16>) outs(%44 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %45 : memref<64x32xf16>
                  %65 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %33)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%65], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                  %66 = loom.bufferize_to_memref %64 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %66, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %30], LR : [%arg5, %30]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<64x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y2y4__d0i2_d1i1_d2i0__f012__dim_y_level0_bc2_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c8 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %22 = arith.muli %20, %c64 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.init_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %27 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %29 = arith.muli %arg3, %c2 : index
                %30 = arith.addi %29, %c1 : index
                loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %29], LR : [%arg5, %30]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %31 = loom.bufferize_to_tensor %27[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %32 = loom.sync ins(%31 : tensor<64x512xf16>) outs(%26 : tensor<64x512xf16>) -> tensor<64x512xf16>
                loom.semaphore_give %27 : memref<64x512xf16>
                %33 = arith.muli %21, %c32 : index
                %34 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %35 = loom.semaphore_take %34 : memref<512x32xf16> -> memref<512x32xf16>
                %36 = loom.init_tensor %35[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.semaphore_take %34 : memref<512x32xf16> -> memref<512x32xf16>
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                %39 = arith.addi %arg4, %29 : index
                loom.copy %reinterpret_cast_0, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %39], LR : [%arg5, %39]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                %40 = loom.bufferize_to_tensor %37[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %41 = loom.sync ins(%40 : tensor<512x32xf16>) outs(%36 : tensor<512x32xf16>) -> tensor<512x32xf16>
                loom.semaphore_give %37 : memref<512x32xf16>
                %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %44 = loom.init_tensor %43[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %47 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %49 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %51 = linalg.fill ins(%cst : f16) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                %52 = linalg.matmul ins(%32, %41 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%51 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %35 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %53 = loom.sync ins(%52 : tensor<64x32xf16>) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %49 : memref<64x32xf16>
                %54 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                %55 = loom.semaphore_take %54 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                %56 = loom.init_tensor %55[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                %57 = loom.semaphore_take %54 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                %59 = loom.gather ins(%53 : tensor<64x32xf16>) outs(%58 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%c0, %39], LR : [%c7, %39]) -> tensor<8x64x32xf16>
                loom.semaphore_give %47 : memref<64x32xf16>
                %60 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %60 {
                  %61 = linalg.fill ins(%cst : f16) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %62 = loom.sync ins(%59 : tensor<8x64x32xf16>) outs(%56 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                  loom.semaphore_give %57 : memref<8x64x32xf16>
                  %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%62 : tensor<8x64x32xf16>) outs(%61 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %67 = arith.addf %in, %out : f16
                    linalg.yield %67 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %55 : memref<8x64x32xf16>
                  %64 = loom.sync ins(%63 : tensor<64x32xf16>) outs(%44 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %45 : memref<64x32xf16>
                  %65 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %33)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%65], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                  %66 = loom.bufferize_to_memref %64 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %66, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %39], LR : [%arg5, %39]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<64x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc4_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c8 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %22 = arith.muli %20, %c64 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.init_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %27 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %29 = arith.muli %arg4, %c4 : index
                %30 = arith.addi %arg3, %29 : index
                loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %30], LR : [%arg5, %30]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %31 = loom.bufferize_to_tensor %27[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %32 = loom.sync ins(%31 : tensor<64x512xf16>) outs(%26 : tensor<64x512xf16>) -> tensor<64x512xf16>
                loom.semaphore_give %27 : memref<64x512xf16>
                %33 = arith.muli %21, %c32 : index
                %34 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %35 = loom.semaphore_take %34 : memref<512x32xf16> -> memref<512x32xf16>
                %36 = loom.init_tensor %35[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.semaphore_take %34 : memref<512x32xf16> -> memref<512x32xf16>
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                %39 = arith.addi %29, %c3 : index
                loom.copy %reinterpret_cast_0, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %29], LR : [%arg5, %39]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                %40 = loom.bufferize_to_tensor %37[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %41 = loom.sync ins(%40 : tensor<512x32xf16>) outs(%36 : tensor<512x32xf16>) -> tensor<512x32xf16>
                loom.semaphore_give %37 : memref<512x32xf16>
                %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %44 = loom.init_tensor %43[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %47 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %49 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %51 = linalg.fill ins(%cst : f16) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                %52 = linalg.matmul ins(%32, %41 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%51 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %35 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %53 = loom.sync ins(%52 : tensor<64x32xf16>) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %49 : memref<64x32xf16>
                %54 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                %55 = loom.semaphore_take %54 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                %56 = loom.init_tensor %55[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                %57 = loom.semaphore_take %54 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                %59 = loom.gather ins(%53 : tensor<64x32xf16>) outs(%58 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%c0, %30], LR : [%c7, %30]) -> tensor<8x64x32xf16>
                loom.semaphore_give %47 : memref<64x32xf16>
                %60 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %60 {
                  %61 = linalg.fill ins(%cst : f16) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %62 = loom.sync ins(%59 : tensor<8x64x32xf16>) outs(%56 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                  loom.semaphore_give %57 : memref<8x64x32xf16>
                  %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%62 : tensor<8x64x32xf16>) outs(%61 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %67 = arith.addf %in, %out : f16
                    linalg.yield %67 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %55 : memref<8x64x32xf16>
                  %64 = loom.sync ins(%63 : tensor<64x32xf16>) outs(%44 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %45 : memref<64x32xf16>
                  %65 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %33)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%65], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                  %66 = loom.bufferize_to_memref %64 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %66, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %30], LR : [%arg5, %30]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<64x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i1_d2i0__f012__dim_y_level0_bc4_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %22 = arith.muli %20, %c64 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.init_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %27 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %29 = arith.muli %arg3, %c4 : index
                %30 = arith.addi %29, %c3 : index
                loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %29], LR : [%arg5, %30]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %31 = loom.bufferize_to_tensor %27[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %32 = loom.sync ins(%31 : tensor<64x512xf16>) outs(%26 : tensor<64x512xf16>) -> tensor<64x512xf16>
                loom.semaphore_give %27 : memref<64x512xf16>
                %33 = arith.muli %21, %c32 : index
                %34 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %35 = loom.semaphore_take %34 : memref<512x32xf16> -> memref<512x32xf16>
                %36 = loom.init_tensor %35[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.semaphore_take %34 : memref<512x32xf16> -> memref<512x32xf16>
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                %39 = arith.addi %arg4, %29 : index
                loom.copy %reinterpret_cast_0, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %39], LR : [%arg5, %39]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                %40 = loom.bufferize_to_tensor %37[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %41 = loom.sync ins(%40 : tensor<512x32xf16>) outs(%36 : tensor<512x32xf16>) -> tensor<512x32xf16>
                loom.semaphore_give %37 : memref<512x32xf16>
                %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %44 = loom.init_tensor %43[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %47 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %49 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %51 = linalg.fill ins(%cst : f16) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                %52 = linalg.matmul ins(%32, %41 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%51 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %35 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %53 = loom.sync ins(%52 : tensor<64x32xf16>) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %49 : memref<64x32xf16>
                %54 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                %55 = loom.semaphore_take %54 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                %56 = loom.init_tensor %55[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                %57 = loom.semaphore_take %54 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                %59 = loom.gather ins(%53 : tensor<64x32xf16>) outs(%58 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%c0, %39], LR : [%c7, %39]) -> tensor<8x64x32xf16>
                loom.semaphore_give %47 : memref<64x32xf16>
                %60 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %60 {
                  %61 = linalg.fill ins(%cst : f16) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %62 = loom.sync ins(%59 : tensor<8x64x32xf16>) outs(%56 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                  loom.semaphore_give %57 : memref<8x64x32xf16>
                  %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%62 : tensor<8x64x32xf16>) outs(%61 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %67 = arith.addf %in, %out : f16
                    linalg.yield %67 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %55 : memref<8x64x32xf16>
                  %64 = loom.sync ins(%63 : tensor<64x32xf16>) outs(%44 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %45 : memref<64x32xf16>
                  %65 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %33)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%65], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                  %66 = loom.bufferize_to_memref %64 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %66, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %39], LR : [%arg5, %39]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<64x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %22 = arith.muli %arg3, %c64 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.init_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %27 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %29 = arith.muli %arg4, %c2 : index
                %30 = arith.addi %arg5, %29 : index
                loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %arg3], LR : [%30, %arg3]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %31 = loom.bufferize_to_tensor %27[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %32 = loom.sync ins(%31 : tensor<64x512xf16>) outs(%26 : tensor<64x512xf16>) -> tensor<64x512xf16>
                loom.semaphore_give %27 : memref<64x512xf16>
                %33 = arith.muli %20, %c32 : index
                %34 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %35 = loom.semaphore_take %34 : memref<512x32xf16> -> memref<512x32xf16>
                %36 = loom.init_tensor %35[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.semaphore_take %34 : memref<512x32xf16> -> memref<512x32xf16>
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%30, %c0], LR : [%30, %c7]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                %39 = loom.bufferize_to_tensor %37[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %40 = loom.sync ins(%39 : tensor<512x32xf16>) outs(%36 : tensor<512x32xf16>) -> tensor<512x32xf16>
                loom.semaphore_give %37 : memref<512x32xf16>
                %41 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %42 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
                %43 = loom.init_tensor %42[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %44 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
                %45 = loom.init_tensor %44[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %46 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
                %47 = loom.init_tensor %46[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %48 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
                %49 = loom.init_tensor %48[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<64x32xf16>) -> tensor<64x32xf16>
                %51 = linalg.matmul ins(%32, %40 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %35 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %52 = loom.sync ins(%51 : tensor<64x32xf16>) outs(%47 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %48 : memref<64x32xf16>
                %53 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                %54 = loom.semaphore_take %53 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                %55 = loom.init_tensor %54[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                %56 = loom.semaphore_take %53 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                %57 = loom.init_tensor %56[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                %58 = arith.addi %29, %c1 : index
                %59 = loom.gather ins(%52 : tensor<64x32xf16>) outs(%57 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%29, %arg3], LR : [%58, %arg3]) -> tensor<8x64x32xf16>
                loom.semaphore_give %46 : memref<64x32xf16>
                %60 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %60 {
                  %61 = linalg.fill ins(%cst : f16) outs(%45 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %62 = loom.sync ins(%59 : tensor<8x64x32xf16>) outs(%55 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                  loom.semaphore_give %56 : memref<8x64x32xf16>
                  %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%62 : tensor<8x64x32xf16>) outs(%61 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %67 = arith.addf %in, %out : f16
                    linalg.yield %67 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %54 : memref<8x64x32xf16>
                  %64 = loom.sync ins(%63 : tensor<64x32xf16>) outs(%43 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %44 : memref<64x32xf16>
                  %65 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %33)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%65], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                  %66 = loom.bufferize_to_memref %64 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %66, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%30, %arg3], LR : [%30, %arg3]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %42 : memref<64x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i1_d2i0__f012__dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                scf.for %arg8 = %c0 to %c4 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                  %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %22 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg8)
                  %23 = arith.muli %20, %c64 : index
                  %24 = arith.muli %22, %c512 : index
                  %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                  %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                  %30 = arith.muli %arg3, %c2 : index
                  %31 = arith.addi %arg5, %30 : index
                  loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%31, %c0], LR : [%31, %c7]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                  %32 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %33 = loom.sync ins(%32 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                  loom.semaphore_give %28 : memref<64x512xf16>
                  %34 = arith.muli %21, %c32 : index
                  %35 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                  %36 = loom.semaphore_take %35 : memref<512x32xf16> -> memref<512x32xf16>
                  %37 = loom.init_tensor %36[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %38 = loom.semaphore_take %35 : memref<512x32xf16> -> memref<512x32xf16>
                  %39 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %34)
                  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%39], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                  loom.copy %reinterpret_cast_0, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %arg4], LR : [%31, %arg4]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                  %40 = loom.bufferize_to_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %41 = loom.sync ins(%40 : tensor<512x32xf16>) outs(%37 : tensor<512x32xf16>) -> tensor<512x32xf16>
                  loom.semaphore_give %38 : memref<512x32xf16>
                  %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                  %44 = loom.init_tensor %43[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                  %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %47 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                  %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %49 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                  %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %51 = linalg.fill ins(%cst : f16) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %52 = linalg.matmul ins(%33, %41 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%51 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %36 : memref<512x32xf16>
                  loom.semaphore_give %26 : memref<64x512xf16>
                  %53 = loom.sync ins(%52 : tensor<64x32xf16>) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %49 : memref<64x32xf16>
                  %54 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                  %55 = loom.semaphore_take %54 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %56 = loom.init_tensor %55[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %57 = loom.semaphore_take %54 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %59 = arith.addi %30, %c1 : index
                  %60 = loom.gather ins(%53 : tensor<64x32xf16>) outs(%58 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%30, %arg4], LR : [%59, %arg4]) -> tensor<8x64x32xf16>
                  loom.semaphore_give %47 : memref<64x32xf16>
                  %61 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %61 {
                    %62 = linalg.fill ins(%cst : f16) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %63 = loom.sync ins(%60 : tensor<8x64x32xf16>) outs(%56 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                    loom.semaphore_give %57 : memref<8x64x32xf16>
                    %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%63 : tensor<8x64x32xf16>) outs(%62 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %68 = arith.addf %in, %out : f16
                      linalg.yield %68 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %55 : memref<8x64x32xf16>
                    %65 = loom.sync ins(%64 : tensor<64x32xf16>) outs(%44 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %45 : memref<64x32xf16>
                    %66 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %34)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%66], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    %67 = loom.bufferize_to_memref %65 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %67, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %arg4], LR : [%31, %arg4]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %43 : memref<64x32xf16>
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
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c8 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %22 = arith.muli %arg3, %c64 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.init_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %27 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %29 = arith.muli %arg4, %c4 : index
                %30 = arith.addi %arg5, %29 : index
                loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %arg3], LR : [%30, %arg3]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %31 = loom.bufferize_to_tensor %27[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %32 = loom.sync ins(%31 : tensor<64x512xf16>) outs(%26 : tensor<64x512xf16>) -> tensor<64x512xf16>
                loom.semaphore_give %27 : memref<64x512xf16>
                %33 = arith.muli %20, %c32 : index
                %34 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %35 = loom.semaphore_take %34 : memref<512x32xf16> -> memref<512x32xf16>
                %36 = loom.init_tensor %35[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.semaphore_take %34 : memref<512x32xf16> -> memref<512x32xf16>
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%30, %c0], LR : [%30, %c7]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                %39 = loom.bufferize_to_tensor %37[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %40 = loom.sync ins(%39 : tensor<512x32xf16>) outs(%36 : tensor<512x32xf16>) -> tensor<512x32xf16>
                loom.semaphore_give %37 : memref<512x32xf16>
                %41 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %42 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
                %43 = loom.init_tensor %42[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %44 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
                %45 = loom.init_tensor %44[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %46 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
                %47 = loom.init_tensor %46[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %48 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
                %49 = loom.init_tensor %48[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %50 = linalg.fill ins(%cst : f16) outs(%49 : tensor<64x32xf16>) -> tensor<64x32xf16>
                %51 = linalg.matmul ins(%32, %40 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %35 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %52 = loom.sync ins(%51 : tensor<64x32xf16>) outs(%47 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %48 : memref<64x32xf16>
                %53 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                %54 = loom.semaphore_take %53 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                %55 = loom.init_tensor %54[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                %56 = loom.semaphore_take %53 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                %57 = loom.init_tensor %56[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                %58 = arith.addi %29, %c3 : index
                %59 = loom.gather ins(%52 : tensor<64x32xf16>) outs(%57 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%29, %arg3], LR : [%58, %arg3]) -> tensor<8x64x32xf16>
                loom.semaphore_give %46 : memref<64x32xf16>
                %60 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %60 {
                  %61 = linalg.fill ins(%cst : f16) outs(%45 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %62 = loom.sync ins(%59 : tensor<8x64x32xf16>) outs(%55 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                  loom.semaphore_give %56 : memref<8x64x32xf16>
                  %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%62 : tensor<8x64x32xf16>) outs(%61 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %67 = arith.addf %in, %out : f16
                    linalg.yield %67 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %54 : memref<8x64x32xf16>
                  %64 = loom.sync ins(%63 : tensor<64x32xf16>) outs(%43 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %44 : memref<64x32xf16>
                  %65 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %33)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%65], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                  %66 = loom.bufferize_to_memref %64 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %66, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%30, %arg3], LR : [%30, %arg3]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %42 : memref<64x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i1_d2i0__f012__dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                scf.for %arg8 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                  %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                  %22 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg8)
                  %23 = arith.muli %20, %c64 : index
                  %24 = arith.muli %22, %c512 : index
                  %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                  %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                  %30 = arith.muli %arg3, %c4 : index
                  %31 = arith.addi %arg5, %30 : index
                  loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%31, %c0], LR : [%31, %c7]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                  %32 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %33 = loom.sync ins(%32 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                  loom.semaphore_give %28 : memref<64x512xf16>
                  %34 = arith.muli %21, %c32 : index
                  %35 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                  %36 = loom.semaphore_take %35 : memref<512x32xf16> -> memref<512x32xf16>
                  %37 = loom.init_tensor %36[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %38 = loom.semaphore_take %35 : memref<512x32xf16> -> memref<512x32xf16>
                  %39 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %34)
                  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%39], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                  loom.copy %reinterpret_cast_0, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %arg4], LR : [%31, %arg4]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                  %40 = loom.bufferize_to_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %41 = loom.sync ins(%40 : tensor<512x32xf16>) outs(%37 : tensor<512x32xf16>) -> tensor<512x32xf16>
                  loom.semaphore_give %38 : memref<512x32xf16>
                  %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                  %44 = loom.init_tensor %43[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                  %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %47 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                  %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %49 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
                  %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %51 = linalg.fill ins(%cst : f16) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %52 = linalg.matmul ins(%33, %41 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%51 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %36 : memref<512x32xf16>
                  loom.semaphore_give %26 : memref<64x512xf16>
                  %53 = loom.sync ins(%52 : tensor<64x32xf16>) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %49 : memref<64x32xf16>
                  %54 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                  %55 = loom.semaphore_take %54 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %56 = loom.init_tensor %55[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %57 = loom.semaphore_take %54 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %59 = arith.addi %30, %c3 : index
                  %60 = loom.gather ins(%53 : tensor<64x32xf16>) outs(%58 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%30, %arg4], LR : [%59, %arg4]) -> tensor<8x64x32xf16>
                  loom.semaphore_give %47 : memref<64x32xf16>
                  %61 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %61 {
                    %62 = linalg.fill ins(%cst : f16) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %63 = loom.sync ins(%60 : tensor<8x64x32xf16>) outs(%56 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                    loom.semaphore_give %57 : memref<8x64x32xf16>
                    %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%63 : tensor<8x64x32xf16>) outs(%62 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %68 = arith.addf %in, %out : f16
                      linalg.yield %68 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %55 : memref<8x64x32xf16>
                    %65 = loom.sync ins(%64 : tensor<64x32xf16>) outs(%44 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %45 : memref<64x32xf16>
                    %66 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %34)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%66], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    %67 = loom.bufferize_to_memref %65 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %67, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %arg4], LR : [%31, %arg4]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %43 : memref<64x32xf16>
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
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y2y2y2__d0i2_d1i2_d2i0_d3i1__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.parallel (%arg6) = (0) to (2) {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                scf.for %arg8 = %c0 to %c8 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %22 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
                  %23 = arith.muli %20, %c64 : index
                  %24 = arith.muli %22, %c512 : index
                  %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                  %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                  %30 = arith.muli %arg3, %c2 : index
                  %31 = arith.addi %arg6, %30 : index
                  %32 = arith.muli %arg4, %c4 : index
                  %33 = arith.addi %31, %32 : index
                  loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                  %34 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %35 = loom.sync ins(%34 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                  loom.semaphore_give %28 : memref<64x512xf16>
                  %36 = arith.muli %21, %c32 : index
                  %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                  %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %39 = loom.init_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %40 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %41 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %36)
                  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                  loom.copy %reinterpret_cast_0, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                  %42 = loom.bufferize_to_tensor %40[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %43 = loom.sync ins(%42 : tensor<512x32xf16>) outs(%39 : tensor<512x32xf16>) -> tensor<512x32xf16>
                  loom.semaphore_give %40 : memref<512x32xf16>
                  %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %49 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %51 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %52 = loom.init_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%52 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %54 = linalg.matmul ins(%35, %43 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %38 : memref<512x32xf16>
                  loom.semaphore_give %26 : memref<64x512xf16>
                  %55 = loom.sync ins(%54 : tensor<64x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %51 : memref<64x32xf16>
                  %56 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                  %57 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %59 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %60 = loom.init_tensor %59[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %61 = loom.gather ins(%55 : tensor<64x32xf16>) outs(%60 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%c0, %33], LR : [%c7, %33]) -> tensor<8x64x32xf16>
                  loom.semaphore_give %49 : memref<64x32xf16>
                  %62 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %62 {
                    %63 = linalg.fill ins(%cst : f16) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %64 = loom.sync ins(%61 : tensor<8x64x32xf16>) outs(%58 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                    loom.semaphore_give %59 : memref<8x64x32xf16>
                    %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%64 : tensor<8x64x32xf16>) outs(%63 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %69 = arith.addf %in, %out : f16
                      linalg.yield %69 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %57 : memref<8x64x32xf16>
                    %66 = loom.sync ins(%65 : tensor<64x32xf16>) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %47 : memref<64x32xf16>
                    %67 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %36)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%67], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    %68 = loom.bufferize_to_memref %66 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %68, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %45 : memref<64x32xf16>
                  }
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y2y2y2__d0i2_d1i2_d2i1_d3i0__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            affine.parallel (%arg6) = (0) to (2) {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                scf.for %arg8 = %c0 to %c8 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %22 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
                  %23 = arith.muli %20, %c64 : index
                  %24 = arith.muli %22, %c512 : index
                  %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                  %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                  %30 = arith.muli %arg4, %c2 : index
                  %31 = arith.addi %arg6, %30 : index
                  %32 = arith.muli %arg3, %c4 : index
                  %33 = arith.addi %31, %32 : index
                  loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                  %34 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %35 = loom.sync ins(%34 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                  loom.semaphore_give %28 : memref<64x512xf16>
                  %36 = arith.muli %21, %c32 : index
                  %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                  %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %39 = loom.init_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %40 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %41 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %36)
                  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                  loom.copy %reinterpret_cast_0, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                  %42 = loom.bufferize_to_tensor %40[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %43 = loom.sync ins(%42 : tensor<512x32xf16>) outs(%39 : tensor<512x32xf16>) -> tensor<512x32xf16>
                  loom.semaphore_give %40 : memref<512x32xf16>
                  %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %49 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %51 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %52 = loom.init_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%52 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %54 = linalg.matmul ins(%35, %43 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %38 : memref<512x32xf16>
                  loom.semaphore_give %26 : memref<64x512xf16>
                  %55 = loom.sync ins(%54 : tensor<64x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %51 : memref<64x32xf16>
                  %56 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                  %57 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %59 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %60 = loom.init_tensor %59[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %61 = loom.gather ins(%55 : tensor<64x32xf16>) outs(%60 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%c0, %33], LR : [%c7, %33]) -> tensor<8x64x32xf16>
                  loom.semaphore_give %49 : memref<64x32xf16>
                  %62 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %62 {
                    %63 = linalg.fill ins(%cst : f16) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %64 = loom.sync ins(%61 : tensor<8x64x32xf16>) outs(%58 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                    loom.semaphore_give %59 : memref<8x64x32xf16>
                    %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%64 : tensor<8x64x32xf16>) outs(%63 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %69 = arith.addf %in, %out : f16
                      linalg.yield %69 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %57 : memref<8x64x32xf16>
                    %66 = loom.sync ins(%65 : tensor<64x32xf16>) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %47 : memref<64x32xf16>
                    %67 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %36)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%67], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    %68 = loom.bufferize_to_memref %66 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %68, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %45 : memref<64x32xf16>
                  }
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y2y4__d0i2_d1i2_d2i0_d3i1__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.parallel (%arg6) = (0) to (2) {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                scf.for %arg8 = %c0 to %c4 step %c1 {
                  scf.for %arg9 = %c0 to %c2 step %c1 {
                    %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                    %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                    %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 4)>(%arg5, %arg6, %arg9)
                    %23 = arith.muli %20, %c64 : index
                    %24 = arith.muli %22, %c512 : index
                    %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                    %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                    %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                    %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                    %30 = arith.muli %arg3, %c2 : index
                    %31 = arith.addi %arg5, %30 : index
                    %32 = arith.muli %arg4, %c2 : index
                    %33 = arith.addi %arg6, %32 : index
                    loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                    %34 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                    %35 = loom.sync ins(%34 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                    loom.semaphore_give %28 : memref<64x512xf16>
                    %36 = arith.muli %21, %c32 : index
                    %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                    %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                    %39 = loom.init_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                    %40 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                    %41 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %36)
                    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                    loom.copy %reinterpret_cast_0, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                    %42 = loom.bufferize_to_tensor %40[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                    %43 = loom.sync ins(%42 : tensor<512x32xf16>) outs(%39 : tensor<512x32xf16>) -> tensor<512x32xf16>
                    loom.semaphore_give %40 : memref<512x32xf16>
                    %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                    %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                    %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %49 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                    %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %51 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                    %52 = loom.init_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %53 = linalg.fill ins(%cst : f16) outs(%52 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %54 = linalg.matmul ins(%35, %43 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %38 : memref<512x32xf16>
                    loom.semaphore_give %26 : memref<64x512xf16>
                    %55 = loom.sync ins(%54 : tensor<64x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %51 : memref<64x32xf16>
                    %56 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                    %57 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                    %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                    %59 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                    %60 = loom.init_tensor %59[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                    %61 = arith.addi %30, %c1 : index
                    %62 = loom.gather ins(%55 : tensor<64x32xf16>) outs(%60 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%30, %33], LR : [%61, %33]) -> tensor<8x64x32xf16>
                    loom.semaphore_give %49 : memref<64x32xf16>
                    %63 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %63 {
                      %64 = linalg.fill ins(%cst : f16) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %65 = loom.sync ins(%62 : tensor<8x64x32xf16>) outs(%58 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                      loom.semaphore_give %59 : memref<8x64x32xf16>
                      %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%65 : tensor<8x64x32xf16>) outs(%64 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %70 = arith.addf %in, %out : f16
                        linalg.yield %70 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %57 : memref<8x64x32xf16>
                      %67 = loom.sync ins(%66 : tensor<64x32xf16>) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %47 : memref<64x32xf16>
                      %68 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %36)
                      %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%68], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                      %69 = loom.bufferize_to_memref %67 : tensor<64x32xf16> -> memref<64x32xf16>
                      loom.copy %69, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %45 : memref<64x32xf16>
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
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y2y4__d0i2_d1i2_d2i1_d3i0__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.parallel (%arg6) = (0) to (2) {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                scf.for %arg8 = %c0 to %c4 step %c1 {
                  scf.for %arg9 = %c0 to %c2 step %c1 {
                    %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                    %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                    %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 4)>(%arg5, %arg6, %arg9)
                    %23 = arith.muli %20, %c64 : index
                    %24 = arith.muli %22, %c512 : index
                    %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                    %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                    %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                    %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                    %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                    %30 = arith.muli %arg4, %c2 : index
                    %31 = arith.addi %arg5, %30 : index
                    %32 = arith.muli %arg3, %c2 : index
                    %33 = arith.addi %arg6, %32 : index
                    loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                    %34 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                    %35 = loom.sync ins(%34 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                    loom.semaphore_give %28 : memref<64x512xf16>
                    %36 = arith.muli %21, %c32 : index
                    %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                    %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                    %39 = loom.init_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                    %40 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                    %41 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %36)
                    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                    loom.copy %reinterpret_cast_0, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                    %42 = loom.bufferize_to_tensor %40[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                    %43 = loom.sync ins(%42 : tensor<512x32xf16>) outs(%39 : tensor<512x32xf16>) -> tensor<512x32xf16>
                    loom.semaphore_give %40 : memref<512x32xf16>
                    %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                    %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                    %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %49 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                    %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %51 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                    %52 = loom.init_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %53 = linalg.fill ins(%cst : f16) outs(%52 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %54 = linalg.matmul ins(%35, %43 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %38 : memref<512x32xf16>
                    loom.semaphore_give %26 : memref<64x512xf16>
                    %55 = loom.sync ins(%54 : tensor<64x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %51 : memref<64x32xf16>
                    %56 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                    %57 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                    %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                    %59 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                    %60 = loom.init_tensor %59[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                    %61 = arith.addi %30, %c1 : index
                    %62 = loom.gather ins(%55 : tensor<64x32xf16>) outs(%60 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%30, %33], LR : [%61, %33]) -> tensor<8x64x32xf16>
                    loom.semaphore_give %49 : memref<64x32xf16>
                    %63 = arith.cmpi eq, %arg5, %c0 : index
                    scf.if %63 {
                      %64 = linalg.fill ins(%cst : f16) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %65 = loom.sync ins(%62 : tensor<8x64x32xf16>) outs(%58 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                      loom.semaphore_give %59 : memref<8x64x32xf16>
                      %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%65 : tensor<8x64x32xf16>) outs(%64 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %out: f16):
                        %70 = arith.addf %in, %out : f16
                        linalg.yield %70 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %57 : memref<8x64x32xf16>
                      %67 = loom.sync ins(%66 : tensor<64x32xf16>) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %47 : memref<64x32xf16>
                      %68 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %36)
                      %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%68], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                      %69 = loom.bufferize_to_memref %67 : tensor<64x32xf16> -> memref<64x32xf16>
                      loom.copy %69, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                      loom.semaphore_give %45 : memref<64x32xf16>
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
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y4y2__d0i2_d1i2_d2i0_d3i1__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.parallel (%arg6) = (0) to (4) {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                scf.for %arg8 = %c0 to %c8 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %22 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg5, %arg6)
                  %23 = arith.muli %20, %c64 : index
                  %24 = arith.muli %22, %c512 : index
                  %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                  %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                  %30 = arith.muli %arg3, %c2 : index
                  %31 = arith.addi %arg5, %30 : index
                  %32 = arith.muli %arg4, %c4 : index
                  %33 = arith.addi %arg6, %32 : index
                  loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                  %34 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %35 = loom.sync ins(%34 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                  loom.semaphore_give %28 : memref<64x512xf16>
                  %36 = arith.muli %21, %c32 : index
                  %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                  %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %39 = loom.init_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %40 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %41 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %36)
                  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                  loom.copy %reinterpret_cast_0, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                  %42 = loom.bufferize_to_tensor %40[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %43 = loom.sync ins(%42 : tensor<512x32xf16>) outs(%39 : tensor<512x32xf16>) -> tensor<512x32xf16>
                  loom.semaphore_give %40 : memref<512x32xf16>
                  %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %49 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %51 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %52 = loom.init_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%52 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %54 = linalg.matmul ins(%35, %43 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %38 : memref<512x32xf16>
                  loom.semaphore_give %26 : memref<64x512xf16>
                  %55 = loom.sync ins(%54 : tensor<64x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %51 : memref<64x32xf16>
                  %56 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                  %57 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %59 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %60 = loom.init_tensor %59[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %61 = arith.addi %30, %c1 : index
                  %62 = loom.gather ins(%55 : tensor<64x32xf16>) outs(%60 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%30, %33], LR : [%61, %33]) -> tensor<8x64x32xf16>
                  loom.semaphore_give %49 : memref<64x32xf16>
                  %63 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %63 {
                    %64 = linalg.fill ins(%cst : f16) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %65 = loom.sync ins(%62 : tensor<8x64x32xf16>) outs(%58 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                    loom.semaphore_give %59 : memref<8x64x32xf16>
                    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%65 : tensor<8x64x32xf16>) outs(%64 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %70 = arith.addf %in, %out : f16
                      linalg.yield %70 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %57 : memref<8x64x32xf16>
                    %67 = loom.sync ins(%66 : tensor<64x32xf16>) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %47 : memref<64x32xf16>
                    %68 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %36)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%68], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    %69 = loom.bufferize_to_memref %67 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %69, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %45 : memref<64x32xf16>
                  }
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y4y2__d0i2_d1i2_d2i1_d3i0__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.parallel (%arg6) = (0) to (4) {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                scf.for %arg8 = %c0 to %c4 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %22 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg5, %arg6)
                  %23 = arith.muli %20, %c64 : index
                  %24 = arith.muli %22, %c512 : index
                  %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                  %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                  %30 = arith.muli %arg4, %c2 : index
                  %31 = arith.addi %arg5, %30 : index
                  %32 = arith.muli %arg3, %c4 : index
                  %33 = arith.addi %arg6, %32 : index
                  loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                  %34 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %35 = loom.sync ins(%34 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                  loom.semaphore_give %28 : memref<64x512xf16>
                  %36 = arith.muli %21, %c32 : index
                  %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                  %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %39 = loom.init_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %40 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %41 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %36)
                  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                  loom.copy %reinterpret_cast_0, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                  %42 = loom.bufferize_to_tensor %40[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %43 = loom.sync ins(%42 : tensor<512x32xf16>) outs(%39 : tensor<512x32xf16>) -> tensor<512x32xf16>
                  loom.semaphore_give %40 : memref<512x32xf16>
                  %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %49 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %51 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %52 = loom.init_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%52 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %54 = linalg.matmul ins(%35, %43 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %38 : memref<512x32xf16>
                  loom.semaphore_give %26 : memref<64x512xf16>
                  %55 = loom.sync ins(%54 : tensor<64x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %51 : memref<64x32xf16>
                  %56 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                  %57 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %59 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %60 = loom.init_tensor %59[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %61 = arith.addi %30, %c1 : index
                  %62 = loom.gather ins(%55 : tensor<64x32xf16>) outs(%60 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%30, %33], LR : [%61, %33]) -> tensor<8x64x32xf16>
                  loom.semaphore_give %49 : memref<64x32xf16>
                  %63 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %63 {
                    %64 = linalg.fill ins(%cst : f16) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %65 = loom.sync ins(%62 : tensor<8x64x32xf16>) outs(%58 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                    loom.semaphore_give %59 : memref<8x64x32xf16>
                    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%65 : tensor<8x64x32xf16>) outs(%64 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %70 = arith.addf %in, %out : f16
                      linalg.yield %70 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %57 : memref<8x64x32xf16>
                    %67 = loom.sync ins(%66 : tensor<64x32xf16>) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %47 : memref<64x32xf16>
                    %68 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %36)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%68], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    %69 = loom.bufferize_to_memref %67 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %69, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %45 : memref<64x32xf16>
                  }
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y2y4__d0i2_d1i2_d2i0_d3i1__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.parallel (%arg6) = (0) to (2) {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                scf.for %arg8 = %c0 to %c4 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg8)
                  %22 = affine.apply affine_map<(d0, d1) -> (d0 * 4 + d1)>(%arg5, %arg6)
                  %23 = arith.muli %20, %c64 : index
                  %24 = arith.muli %22, %c512 : index
                  %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                  %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                  %30 = arith.muli %arg3, %c4 : index
                  %31 = arith.addi %arg5, %30 : index
                  %32 = arith.muli %arg4, %c2 : index
                  %33 = arith.addi %arg6, %32 : index
                  loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                  %34 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %35 = loom.sync ins(%34 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                  loom.semaphore_give %28 : memref<64x512xf16>
                  %36 = arith.muli %21, %c32 : index
                  %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                  %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %39 = loom.init_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %40 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %41 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %36)
                  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                  loom.copy %reinterpret_cast_0, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                  %42 = loom.bufferize_to_tensor %40[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %43 = loom.sync ins(%42 : tensor<512x32xf16>) outs(%39 : tensor<512x32xf16>) -> tensor<512x32xf16>
                  loom.semaphore_give %40 : memref<512x32xf16>
                  %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %49 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %51 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %52 = loom.init_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%52 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %54 = linalg.matmul ins(%35, %43 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %38 : memref<512x32xf16>
                  loom.semaphore_give %26 : memref<64x512xf16>
                  %55 = loom.sync ins(%54 : tensor<64x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %51 : memref<64x32xf16>
                  %56 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                  %57 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %59 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %60 = loom.init_tensor %59[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %61 = arith.addi %30, %c3 : index
                  %62 = loom.gather ins(%55 : tensor<64x32xf16>) outs(%60 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%30, %33], LR : [%61, %33]) -> tensor<8x64x32xf16>
                  loom.semaphore_give %49 : memref<64x32xf16>
                  %63 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %63 {
                    %64 = linalg.fill ins(%cst : f16) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %65 = loom.sync ins(%62 : tensor<8x64x32xf16>) outs(%58 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                    loom.semaphore_give %59 : memref<8x64x32xf16>
                    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%65 : tensor<8x64x32xf16>) outs(%64 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %70 = arith.addf %in, %out : f16
                      linalg.yield %70 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %57 : memref<8x64x32xf16>
                    %67 = loom.sync ins(%66 : tensor<64x32xf16>) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %47 : memref<64x32xf16>
                    %68 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %36)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%68], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    %69 = loom.bufferize_to_memref %67 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %69, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %45 : memref<64x32xf16>
                  }
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y2y4__d0i2_d1i2_d2i1_d3i0__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c8 = arith.constant 8 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.parallel (%arg6) = (0) to (2) {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                scf.for %arg8 = %c0 to %c8 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg7)
                  %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %22 = affine.apply affine_map<(d0, d1) -> (d0 * 4 + d1)>(%arg5, %arg6)
                  %23 = arith.muli %20, %c64 : index
                  %24 = arith.muli %22, %c512 : index
                  %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                  %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                  %30 = arith.muli %arg4, %c4 : index
                  %31 = arith.addi %arg5, %30 : index
                  %32 = arith.muli %arg3, %c2 : index
                  %33 = arith.addi %arg6, %32 : index
                  loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                  %34 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %35 = loom.sync ins(%34 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                  loom.semaphore_give %28 : memref<64x512xf16>
                  %36 = arith.muli %21, %c32 : index
                  %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                  %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %39 = loom.init_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %40 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %41 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %36)
                  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                  loom.copy %reinterpret_cast_0, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                  %42 = loom.bufferize_to_tensor %40[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %43 = loom.sync ins(%42 : tensor<512x32xf16>) outs(%39 : tensor<512x32xf16>) -> tensor<512x32xf16>
                  loom.semaphore_give %40 : memref<512x32xf16>
                  %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %49 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %51 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %52 = loom.init_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%52 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %54 = linalg.matmul ins(%35, %43 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %38 : memref<512x32xf16>
                  loom.semaphore_give %26 : memref<64x512xf16>
                  %55 = loom.sync ins(%54 : tensor<64x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %51 : memref<64x32xf16>
                  %56 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                  %57 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %59 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %60 = loom.init_tensor %59[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %61 = arith.addi %30, %c3 : index
                  %62 = loom.gather ins(%55 : tensor<64x32xf16>) outs(%60 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%30, %33], LR : [%61, %33]) -> tensor<8x64x32xf16>
                  loom.semaphore_give %49 : memref<64x32xf16>
                  %63 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %63 {
                    %64 = linalg.fill ins(%cst : f16) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %65 = loom.sync ins(%62 : tensor<8x64x32xf16>) outs(%58 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                    loom.semaphore_give %59 : memref<8x64x32xf16>
                    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%65 : tensor<8x64x32xf16>) outs(%64 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %70 = arith.addf %in, %out : f16
                      linalg.yield %70 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %57 : memref<8x64x32xf16>
                    %67 = loom.sync ins(%66 : tensor<64x32xf16>) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %47 : memref<64x32xf16>
                    %68 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %36)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%68], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    %69 = loom.bufferize_to_memref %67 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %69, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %45 : memref<64x32xf16>
                  }
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y4y2__d0i2_d1i2_d2i0_d3i1__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c8 = arith.constant 8 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.parallel (%arg6) = (0) to (4) {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                scf.for %arg8 = %c0 to %c8 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %22 = affine.apply affine_map<(d0, d1) -> (d0 * 4 + d1)>(%arg5, %arg6)
                  %23 = arith.muli %20, %c64 : index
                  %24 = arith.muli %22, %c512 : index
                  %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                  %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                  %30 = arith.muli %arg3, %c4 : index
                  %31 = arith.addi %arg5, %30 : index
                  %32 = arith.muli %arg4, %c4 : index
                  %33 = arith.addi %arg6, %32 : index
                  loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                  %34 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %35 = loom.sync ins(%34 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                  loom.semaphore_give %28 : memref<64x512xf16>
                  %36 = arith.muli %21, %c32 : index
                  %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                  %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %39 = loom.init_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %40 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %41 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %36)
                  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                  loom.copy %reinterpret_cast_0, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                  %42 = loom.bufferize_to_tensor %40[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %43 = loom.sync ins(%42 : tensor<512x32xf16>) outs(%39 : tensor<512x32xf16>) -> tensor<512x32xf16>
                  loom.semaphore_give %40 : memref<512x32xf16>
                  %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %49 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %51 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %52 = loom.init_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%52 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %54 = linalg.matmul ins(%35, %43 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %38 : memref<512x32xf16>
                  loom.semaphore_give %26 : memref<64x512xf16>
                  %55 = loom.sync ins(%54 : tensor<64x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %51 : memref<64x32xf16>
                  %56 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                  %57 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %59 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %60 = loom.init_tensor %59[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %61 = arith.addi %30, %c3 : index
                  %62 = loom.gather ins(%55 : tensor<64x32xf16>) outs(%60 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%30, %33], LR : [%61, %33]) -> tensor<8x64x32xf16>
                  loom.semaphore_give %49 : memref<64x32xf16>
                  %63 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %63 {
                    %64 = linalg.fill ins(%cst : f16) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %65 = loom.sync ins(%62 : tensor<8x64x32xf16>) outs(%58 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                    loom.semaphore_give %59 : memref<8x64x32xf16>
                    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%65 : tensor<8x64x32xf16>) outs(%64 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %70 = arith.addf %in, %out : f16
                      linalg.yield %70 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %57 : memref<8x64x32xf16>
                    %67 = loom.sync ins(%66 : tensor<64x32xf16>) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %47 : memref<64x32xf16>
                    %68 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %36)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%68], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    %69 = loom.bufferize_to_memref %67 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %69, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %45 : memref<64x32xf16>
                  }
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y4y2__d0i2_d1i2_d2i1_d3i0__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c8 = arith.constant 8 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            affine.parallel (%arg6) = (0) to (4) {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                scf.for %arg8 = %c0 to %c8 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %22 = affine.apply affine_map<(d0, d1) -> (d0 * 4 + d1)>(%arg5, %arg6)
                  %23 = arith.muli %20, %c64 : index
                  %24 = arith.muli %22, %c512 : index
                  %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                  %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                  %30 = arith.muli %arg4, %c4 : index
                  %31 = arith.addi %arg5, %30 : index
                  %32 = arith.muli %arg3, %c4 : index
                  %33 = arith.addi %arg6, %32 : index
                  loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                  %34 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %35 = loom.sync ins(%34 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                  loom.semaphore_give %28 : memref<64x512xf16>
                  %36 = arith.muli %21, %c32 : index
                  %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                  %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %39 = loom.init_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %40 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %41 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %36)
                  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                  loom.copy %reinterpret_cast_0, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                  %42 = loom.bufferize_to_tensor %40[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %43 = loom.sync ins(%42 : tensor<512x32xf16>) outs(%39 : tensor<512x32xf16>) -> tensor<512x32xf16>
                  loom.semaphore_give %40 : memref<512x32xf16>
                  %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %49 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %51 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %52 = loom.init_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%52 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %54 = linalg.matmul ins(%35, %43 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %38 : memref<512x32xf16>
                  loom.semaphore_give %26 : memref<64x512xf16>
                  %55 = loom.sync ins(%54 : tensor<64x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %51 : memref<64x32xf16>
                  %56 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                  %57 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %59 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %60 = loom.init_tensor %59[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %61 = arith.addi %30, %c3 : index
                  %62 = loom.gather ins(%55 : tensor<64x32xf16>) outs(%60 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%30, %33], LR : [%61, %33]) -> tensor<8x64x32xf16>
                  loom.semaphore_give %49 : memref<64x32xf16>
                  %63 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %63 {
                    %64 = linalg.fill ins(%cst : f16) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %65 = loom.sync ins(%62 : tensor<8x64x32xf16>) outs(%58 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                    loom.semaphore_give %59 : memref<8x64x32xf16>
                    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%65 : tensor<8x64x32xf16>) outs(%64 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %70 = arith.addf %in, %out : f16
                      linalg.yield %70 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %57 : memref<8x64x32xf16>
                    %67 = loom.sync ins(%66 : tensor<64x32xf16>) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %47 : memref<64x32xf16>
                    %68 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %36)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%68], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    %69 = loom.bufferize_to_memref %67 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %69, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %33], LR : [%31, %33]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %45 : memref<64x32xf16>
                  }
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x2x2_y8__d0i2_d1i2_d2i0_d3i1__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.parallel (%arg6) = (0) to (8) {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                scf.for %arg8 = %c0 to %c8 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %22 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg5, %arg6)
                  %23 = arith.muli %20, %c64 : index
                  %24 = arith.muli %22, %c512 : index
                  %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                  %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                  %30 = arith.muli %arg3, %c2 : index
                  %31 = arith.addi %arg5, %30 : index
                  %32 = arith.muli %arg4, %c4 : index
                  %33 = arith.addi %31, %32 : index
                  loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                  %34 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %35 = loom.sync ins(%34 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                  loom.semaphore_give %28 : memref<64x512xf16>
                  %36 = arith.muli %21, %c32 : index
                  %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                  %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %39 = loom.init_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %40 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %41 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %36)
                  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                  loom.copy %reinterpret_cast_0, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                  %42 = loom.bufferize_to_tensor %40[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %43 = loom.sync ins(%42 : tensor<512x32xf16>) outs(%39 : tensor<512x32xf16>) -> tensor<512x32xf16>
                  loom.semaphore_give %40 : memref<512x32xf16>
                  %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %49 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %51 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %52 = loom.init_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%52 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %54 = linalg.matmul ins(%35, %43 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %38 : memref<512x32xf16>
                  loom.semaphore_give %26 : memref<64x512xf16>
                  %55 = loom.sync ins(%54 : tensor<64x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %51 : memref<64x32xf16>
                  %56 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                  %57 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %59 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %60 = loom.init_tensor %59[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %61 = arith.addi %30, %32 : index
                  %62 = arith.addi %30, %c1 : index
                  %63 = arith.addi %62, %32 : index
                  %64 = loom.gather ins(%55 : tensor<64x32xf16>) outs(%60 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%61, %arg6], LR : [%63, %arg6]) -> tensor<8x64x32xf16>
                  loom.semaphore_give %49 : memref<64x32xf16>
                  %65 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %65 {
                    %66 = linalg.fill ins(%cst : f16) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %67 = loom.sync ins(%64 : tensor<8x64x32xf16>) outs(%58 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                    loom.semaphore_give %59 : memref<8x64x32xf16>
                    %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%67 : tensor<8x64x32xf16>) outs(%66 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %72 = arith.addf %in, %out : f16
                      linalg.yield %72 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %57 : memref<8x64x32xf16>
                    %69 = loom.sync ins(%68 : tensor<64x32xf16>) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %47 : memref<64x32xf16>
                    %70 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %36)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%70], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    %71 = loom.bufferize_to_memref %69 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %71, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %45 : memref<64x32xf16>
                  }
                } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x2x2_y8__d0i2_d1i2_d2i1_d3i0__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (2) {
            affine.parallel (%arg6) = (0) to (8) {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                scf.for %arg8 = %c0 to %c8 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg7)
                  %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg8)
                  %22 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg5, %arg6)
                  %23 = arith.muli %20, %c64 : index
                  %24 = arith.muli %22, %c512 : index
                  %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                  %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %27 = loom.init_tensor %26[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %28 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
                  %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
                  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                  %30 = arith.muli %arg4, %c2 : index
                  %31 = arith.addi %arg5, %30 : index
                  %32 = arith.muli %arg3, %c4 : index
                  %33 = arith.addi %31, %32 : index
                  loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                  %34 = loom.bufferize_to_tensor %28[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                  %35 = loom.sync ins(%34 : tensor<64x512xf16>) outs(%27 : tensor<64x512xf16>) -> tensor<64x512xf16>
                  loom.semaphore_give %28 : memref<64x512xf16>
                  %36 = arith.muli %21, %c32 : index
                  %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                  %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %39 = loom.init_tensor %38[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %40 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
                  %41 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%24, %36)
                  %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
                  loom.copy %reinterpret_cast_0, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
                  %42 = loom.bufferize_to_tensor %40[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                  %43 = loom.sync ins(%42 : tensor<512x32xf16>) outs(%39 : tensor<512x32xf16>) -> tensor<512x32xf16>
                  loom.semaphore_give %40 : memref<512x32xf16>
                  %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %46 = loom.init_tensor %45[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %48 = loom.init_tensor %47[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %49 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %51 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
                  %52 = loom.init_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%52 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %54 = linalg.matmul ins(%35, %43 : tensor<64x512xf16>, tensor<512x32xf16>) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %38 : memref<512x32xf16>
                  loom.semaphore_give %26 : memref<64x512xf16>
                  %55 = loom.sync ins(%54 : tensor<64x32xf16>) outs(%50 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %51 : memref<64x32xf16>
                  %56 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
                  %57 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %58 = loom.init_tensor %57[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %59 = loom.semaphore_take %56 : memref<8x64x32xf16> -> memref<8x64x32xf16>
                  %60 = loom.init_tensor %59[8, 64, 32] : memref<8x64x32xf16> -> tensor<8x64x32xf16>
                  %61 = arith.addi %30, %32 : index
                  %62 = arith.addi %30, %c1 : index
                  %63 = arith.addi %62, %32 : index
                  %64 = loom.gather ins(%55 : tensor<64x32xf16>) outs(%60 : tensor<8x64x32xf16>) across(%arg5 : index) region : (UL : [%61, %arg6], LR : [%63, %arg6]) -> tensor<8x64x32xf16>
                  loom.semaphore_give %49 : memref<64x32xf16>
                  %65 = arith.cmpi eq, %arg5, %c0 : index
                  scf.if %65 {
                    %66 = linalg.fill ins(%cst : f16) outs(%48 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %67 = loom.sync ins(%64 : tensor<8x64x32xf16>) outs(%58 : tensor<8x64x32xf16>) -> tensor<8x64x32xf16>
                    loom.semaphore_give %59 : memref<8x64x32xf16>
                    %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%67 : tensor<8x64x32xf16>) outs(%66 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %72 = arith.addf %in, %out : f16
                      linalg.yield %72 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %57 : memref<8x64x32xf16>
                    %69 = loom.sync ins(%68 : tensor<64x32xf16>) outs(%46 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %47 : memref<64x32xf16>
                    %70 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %36)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%70], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    %71 = loom.bufferize_to_memref %69 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %71, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                    loom.semaphore_give %45 : memref<64x32xf16>
                  }
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
