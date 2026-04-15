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
    func.func @split_k_matmul_gather__x8_y2y4__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc2_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %22 = arith.muli %20, %c64 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %27 = arith.muli %arg4, %c2 : index
                %28 = arith.addi %arg3, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %29 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %30 = arith.muli %21, %c64 : index
                %31 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %32 = loom.semaphore_take %31 : memref<512x64xf16> -> memref<512x64xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
                %34 = arith.addi %27, %c1 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %27], LR : [%arg5, %34]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
                %35 = loom.bufferize_to_tensor %32[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %38 = loom.init_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %39 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %40 = loom.init_tensor %39[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = linalg.fill ins(%cst : f16) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %42 = linalg.matmul ins(%29, %35 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%41 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %32 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %43 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %43 {
                  %44 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
                  %45 = loom.semaphore_take %44 : memref<8x64x64xf16> -> memref<8x64x64xf16>
                  %46 = loom.init_tensor %45[8, 64, 64] : memref<8x64x64xf16> -> tensor<8x64x64xf16>
                  %47 = loom.gather ins(%42 : tensor<64x64xf16>) outs(%46 : tensor<8x64x64xf16>) across(%arg5 : index) region : (UL : [%c0, %28], LR : [%c7, %28]) -> tensor<8x64x64xf16>
                  loom.semaphore_give %39 : memref<64x64xf16>
                  %48 = linalg.fill ins(%cst : f16) outs(%38 : tensor<64x64xf16>) -> tensor<64x64xf16>
                  %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%47 : tensor<8x64x64xf16>) outs(%48 : tensor<64x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %52 = arith.addf %in, %out : f16
                    linalg.yield %52 : f16
                  } -> tensor<64x64xf16>
                  loom.semaphore_give %45 : memref<8x64x64xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<64x64xf16> -> memref<64x64xf16>
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %37 : memref<64x64xf16>
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
    func.func @split_k_matmul_gather__x8_y2y4__d0i2_d1i1_d2i0__f012__dim_y_level0_bc2_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %22 = arith.muli %20, %c64 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %27 = arith.muli %arg3, %c2 : index
                %28 = arith.addi %27, %c1 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %27], LR : [%arg5, %28]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %29 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %30 = arith.muli %21, %c64 : index
                %31 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %32 = loom.semaphore_take %31 : memref<512x64xf16> -> memref<512x64xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
                %34 = arith.addi %arg4, %27 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %34], LR : [%arg5, %34]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
                %35 = loom.bufferize_to_tensor %32[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %38 = loom.init_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %39 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %40 = loom.init_tensor %39[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = linalg.fill ins(%cst : f16) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %42 = linalg.matmul ins(%29, %35 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%41 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %32 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %43 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %43 {
                  %44 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
                  %45 = loom.semaphore_take %44 : memref<8x64x64xf16> -> memref<8x64x64xf16>
                  %46 = loom.init_tensor %45[8, 64, 64] : memref<8x64x64xf16> -> tensor<8x64x64xf16>
                  %47 = loom.gather ins(%42 : tensor<64x64xf16>) outs(%46 : tensor<8x64x64xf16>) across(%arg5 : index) region : (UL : [%c0, %34], LR : [%c7, %34]) -> tensor<8x64x64xf16>
                  loom.semaphore_give %39 : memref<64x64xf16>
                  %48 = linalg.fill ins(%cst : f16) outs(%38 : tensor<64x64xf16>) -> tensor<64x64xf16>
                  %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%47 : tensor<8x64x64xf16>) outs(%48 : tensor<64x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %52 = arith.addf %in, %out : f16
                    linalg.yield %52 : f16
                  } -> tensor<64x64xf16>
                  loom.semaphore_give %45 : memref<8x64x64xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<64x64xf16> -> memref<64x64xf16>
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %34], LR : [%arg5, %34]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %37 : memref<64x64xf16>
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
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc4_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %22 = arith.muli %20, %c64 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %27 = arith.muli %arg4, %c4 : index
                %28 = arith.addi %arg3, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %29 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %30 = arith.muli %21, %c64 : index
                %31 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %32 = loom.semaphore_take %31 : memref<512x64xf16> -> memref<512x64xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
                %34 = arith.addi %27, %c3 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %27], LR : [%arg5, %34]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
                %35 = loom.bufferize_to_tensor %32[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %38 = loom.init_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %39 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %40 = loom.init_tensor %39[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = linalg.fill ins(%cst : f16) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %42 = linalg.matmul ins(%29, %35 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%41 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %32 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %43 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %43 {
                  %44 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
                  %45 = loom.semaphore_take %44 : memref<8x64x64xf16> -> memref<8x64x64xf16>
                  %46 = loom.init_tensor %45[8, 64, 64] : memref<8x64x64xf16> -> tensor<8x64x64xf16>
                  %47 = loom.gather ins(%42 : tensor<64x64xf16>) outs(%46 : tensor<8x64x64xf16>) across(%arg5 : index) region : (UL : [%c0, %28], LR : [%c7, %28]) -> tensor<8x64x64xf16>
                  loom.semaphore_give %39 : memref<64x64xf16>
                  %48 = linalg.fill ins(%cst : f16) outs(%38 : tensor<64x64xf16>) -> tensor<64x64xf16>
                  %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%47 : tensor<8x64x64xf16>) outs(%48 : tensor<64x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %52 = arith.addf %in, %out : f16
                    linalg.yield %52 : f16
                  } -> tensor<64x64xf16>
                  loom.semaphore_give %45 : memref<8x64x64xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<64x64xf16> -> memref<64x64xf16>
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %37 : memref<64x64xf16>
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
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i1_d2i0__f012__dim_y_level0_bc4_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %22 = arith.muli %20, %c64 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %27 = arith.muli %arg3, %c4 : index
                %28 = arith.addi %27, %c3 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %27], LR : [%arg5, %28]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %29 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %30 = arith.muli %21, %c64 : index
                %31 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %32 = loom.semaphore_take %31 : memref<512x64xf16> -> memref<512x64xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
                %34 = arith.addi %arg4, %27 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %34], LR : [%arg5, %34]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
                %35 = loom.bufferize_to_tensor %32[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %38 = loom.init_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %39 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %40 = loom.init_tensor %39[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = linalg.fill ins(%cst : f16) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %42 = linalg.matmul ins(%29, %35 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%41 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %32 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %43 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %43 {
                  %44 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
                  %45 = loom.semaphore_take %44 : memref<8x64x64xf16> -> memref<8x64x64xf16>
                  %46 = loom.init_tensor %45[8, 64, 64] : memref<8x64x64xf16> -> tensor<8x64x64xf16>
                  %47 = loom.gather ins(%42 : tensor<64x64xf16>) outs(%46 : tensor<8x64x64xf16>) across(%arg5 : index) region : (UL : [%c0, %34], LR : [%c7, %34]) -> tensor<8x64x64xf16>
                  loom.semaphore_give %39 : memref<64x64xf16>
                  %48 = linalg.fill ins(%cst : f16) outs(%38 : tensor<64x64xf16>) -> tensor<64x64xf16>
                  %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%47 : tensor<8x64x64xf16>) outs(%48 : tensor<64x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %52 = arith.addf %in, %out : f16
                    linalg.yield %52 : f16
                  } -> tensor<64x64xf16>
                  loom.semaphore_give %45 : memref<8x64x64xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<64x64xf16> -> memref<64x64xf16>
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %34], LR : [%arg5, %34]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %37 : memref<64x64xf16>
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
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %22 = arith.muli %arg3, %c64 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %27 = arith.muli %arg4, %c2 : index
                %28 = arith.addi %arg5, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %arg3], LR : [%28, %arg3]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %29 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %30 = arith.muli %20, %c64 : index
                %31 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %32 = loom.semaphore_take %31 : memref<512x64xf16> -> memref<512x64xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%28, %c0], LR : [%28, %c7]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
                %34 = loom.bufferize_to_tensor %32[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %35 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %36 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
                %37 = loom.init_tensor %36[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %38 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
                %39 = loom.init_tensor %38[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %41 = linalg.matmul ins(%29, %34 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %32 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %42 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %42 {
                  %43 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
                  %44 = loom.semaphore_take %43 : memref<8x64x64xf16> -> memref<8x64x64xf16>
                  %45 = loom.init_tensor %44[8, 64, 64] : memref<8x64x64xf16> -> tensor<8x64x64xf16>
                  %46 = arith.addi %27, %c1 : index
                  %47 = loom.gather ins(%41 : tensor<64x64xf16>) outs(%45 : tensor<8x64x64xf16>) across(%arg5 : index) region : (UL : [%27, %arg3], LR : [%46, %arg3]) -> tensor<8x64x64xf16>
                  loom.semaphore_give %38 : memref<64x64xf16>
                  %48 = linalg.fill ins(%cst : f16) outs(%37 : tensor<64x64xf16>) -> tensor<64x64xf16>
                  %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%47 : tensor<8x64x64xf16>) outs(%48 : tensor<64x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %52 = arith.addf %in, %out : f16
                    linalg.yield %52 : f16
                  } -> tensor<64x64xf16>
                  loom.semaphore_give %44 : memref<8x64x64xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<64x64xf16> -> memref<64x64xf16>
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%28, %arg3], LR : [%28, %arg3]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %36 : memref<64x64xf16>
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
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i1_d2i0__f012__dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %22 = arith.muli %20, %c64 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %27 = arith.muli %arg3, %c2 : index
                %28 = arith.addi %arg5, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%28, %c0], LR : [%28, %c7]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %29 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %30 = arith.muli %arg4, %c64 : index
                %31 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %32 = loom.semaphore_take %31 : memref<512x64xf16> -> memref<512x64xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %arg4], LR : [%28, %arg4]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
                %34 = loom.bufferize_to_tensor %32[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %35 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %36 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
                %37 = loom.init_tensor %36[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %38 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
                %39 = loom.init_tensor %38[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %41 = linalg.matmul ins(%29, %34 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %32 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %42 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %42 {
                  %43 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
                  %44 = loom.semaphore_take %43 : memref<8x64x64xf16> -> memref<8x64x64xf16>
                  %45 = loom.init_tensor %44[8, 64, 64] : memref<8x64x64xf16> -> tensor<8x64x64xf16>
                  %46 = arith.addi %27, %c1 : index
                  %47 = loom.gather ins(%41 : tensor<64x64xf16>) outs(%45 : tensor<8x64x64xf16>) across(%arg5 : index) region : (UL : [%27, %arg4], LR : [%46, %arg4]) -> tensor<8x64x64xf16>
                  loom.semaphore_give %38 : memref<64x64xf16>
                  %48 = linalg.fill ins(%cst : f16) outs(%37 : tensor<64x64xf16>) -> tensor<64x64xf16>
                  %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%47 : tensor<8x64x64xf16>) outs(%48 : tensor<64x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %52 = arith.addf %in, %out : f16
                    linalg.yield %52 : f16
                  } -> tensor<64x64xf16>
                  loom.semaphore_give %44 : memref<8x64x64xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<64x64xf16> -> memref<64x64xf16>
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%28, %arg4], LR : [%28, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %36 : memref<64x64xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %22 = arith.muli %arg3, %c64 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %27 = arith.muli %arg4, %c4 : index
                %28 = arith.addi %arg5, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %arg3], LR : [%28, %arg3]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %29 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %30 = arith.muli %20, %c64 : index
                %31 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %32 = loom.semaphore_take %31 : memref<512x64xf16> -> memref<512x64xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%28, %c0], LR : [%28, %c7]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
                %34 = loom.bufferize_to_tensor %32[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %35 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %36 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
                %37 = loom.init_tensor %36[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %38 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
                %39 = loom.init_tensor %38[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %41 = linalg.matmul ins(%29, %34 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %32 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %42 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %42 {
                  %43 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
                  %44 = loom.semaphore_take %43 : memref<8x64x64xf16> -> memref<8x64x64xf16>
                  %45 = loom.init_tensor %44[8, 64, 64] : memref<8x64x64xf16> -> tensor<8x64x64xf16>
                  %46 = arith.addi %27, %c3 : index
                  %47 = loom.gather ins(%41 : tensor<64x64xf16>) outs(%45 : tensor<8x64x64xf16>) across(%arg5 : index) region : (UL : [%27, %arg3], LR : [%46, %arg3]) -> tensor<8x64x64xf16>
                  loom.semaphore_give %38 : memref<64x64xf16>
                  %48 = linalg.fill ins(%cst : f16) outs(%37 : tensor<64x64xf16>) -> tensor<64x64xf16>
                  %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%47 : tensor<8x64x64xf16>) outs(%48 : tensor<64x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %52 = arith.addf %in, %out : f16
                    linalg.yield %52 : f16
                  } -> tensor<64x64xf16>
                  loom.semaphore_give %44 : memref<8x64x64xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<64x64xf16> -> memref<64x64xf16>
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%28, %arg3], LR : [%28, %arg3]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %36 : memref<64x64xf16>
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
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i1_d2i0__f012__dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %22 = arith.muli %20, %c64 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
                %27 = arith.muli %arg3, %c4 : index
                %28 = arith.addi %arg5, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%28, %c0], LR : [%28, %c7]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
                %29 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %30 = arith.muli %arg4, %c64 : index
                %31 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %32 = loom.semaphore_take %31 : memref<512x64xf16> -> memref<512x64xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %arg4], LR : [%28, %arg4]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
                %34 = loom.bufferize_to_tensor %32[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %35 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %36 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
                %37 = loom.init_tensor %36[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %38 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
                %39 = loom.init_tensor %38[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %41 = linalg.matmul ins(%29, %34 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %32 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %42 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %42 {
                  %43 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
                  %44 = loom.semaphore_take %43 : memref<8x64x64xf16> -> memref<8x64x64xf16>
                  %45 = loom.init_tensor %44[8, 64, 64] : memref<8x64x64xf16> -> tensor<8x64x64xf16>
                  %46 = arith.addi %27, %c3 : index
                  %47 = loom.gather ins(%41 : tensor<64x64xf16>) outs(%45 : tensor<8x64x64xf16>) across(%arg5 : index) region : (UL : [%27, %arg4], LR : [%46, %arg4]) -> tensor<8x64x64xf16>
                  loom.semaphore_give %38 : memref<64x64xf16>
                  %48 = linalg.fill ins(%cst : f16) outs(%37 : tensor<64x64xf16>) -> tensor<64x64xf16>
                  %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%47 : tensor<8x64x64xf16>) outs(%48 : tensor<64x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %52 = arith.addf %in, %out : f16
                    linalg.yield %52 : f16
                  } -> tensor<64x64xf16>
                  loom.semaphore_give %44 : memref<8x64x64xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<64x64xf16> -> memref<64x64xf16>
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%28, %arg4], LR : [%28, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                  loom.semaphore_give %36 : memref<64x64xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
