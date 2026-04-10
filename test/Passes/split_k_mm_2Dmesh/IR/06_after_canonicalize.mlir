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
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d1i0_d2i1_d0i2__f012__n_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c2 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d1i0_d2i1_d0i2__f012__n_n_dim_x_level0_bc8__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c2 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d1i0_d2i1_d0i2__f012__n_dim_y_level0_bc2_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c2 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d1i0_d2i1_d0i2__f012__n_dim_y_level0_bc2_dim_x_level0_bc8__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c2 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012__n_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c4 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012__n_n_dim_x_level0_bc8__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c4 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012__dim_y_level0_bc2_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c4 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012__dim_y_level0_bc2_n_dim_x_level0_bc8__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c4 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d1i0_d2i1_d0i2__f012__n_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c4 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d1i0_d2i1_d0i2__f012__n_n_dim_x_level0_bc8__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c4 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d1i0_d2i1_d0i2__f012__n_dim_y_level0_bc4_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c4 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d1i0_d2i1_d0i2__f012__n_dim_y_level0_bc4_dim_x_level0_bc8__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c4 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012__n_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c2 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012__n_n_dim_x_level0_bc8__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c2 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012__dim_y_level0_bc4_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c2 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012__dim_y_level0_bc4_n_dim_x_level0_bc8__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %arg5, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %21, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c2 : index
                  %44 = arith.addi %arg3, %43 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%c0, %44], LR : [%c7, %44]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d1i0_d2i1_d0i2__f012__n_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %22 = arith.muli %arg3, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %20, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c2 : index
                  %44 = arith.addi %43, %c1 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%43, %arg3], LR : [%44, %arg3]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d1i0_d2i1_d0i2__f012__n_n_dim_x_level0_bc2__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %22 = arith.muli %arg3, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %20, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c2 : index
                  %44 = arith.addi %43, %c1 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%43, %arg3], LR : [%44, %arg3]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d1i0_d2i1_d0i2__f012__n_dim_y_level0_bc8_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %22 = arith.muli %arg3, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %20, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c2 : index
                  %44 = arith.addi %43, %c1 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%43, %arg3], LR : [%44, %arg3]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d1i0_d2i1_d0i2__f012__n_dim_y_level0_bc8_dim_x_level0_bc2__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %22 = arith.muli %arg3, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %20, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c2 : index
                  %44 = arith.addi %43, %c1 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%43, %arg3], LR : [%44, %arg3]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012__n_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %arg4, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.addi %arg3, %c4 : index
                  %44 = loom.reduce_sum ins(%37) outs(%42) (UL : [%arg3, %arg4], LR : [%43, %arg4]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %45 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %44 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %46, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012__n_n_dim_x_level0_bc2__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %arg4, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.addi %arg3, %c4 : index
                  %44 = loom.reduce_sum ins(%37) outs(%42) (UL : [%arg3, %arg4], LR : [%43, %arg4]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %45 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %44 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %46, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012__dim_y_level0_bc8_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %arg4, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.addi %arg3, %c4 : index
                  %44 = loom.reduce_sum ins(%37) outs(%42) (UL : [%arg3, %arg4], LR : [%43, %arg4]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %45 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %44 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %46, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012__dim_y_level0_bc8_n_dim_x_level0_bc2__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %arg4, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.addi %arg3, %c4 : index
                  %44 = loom.reduce_sum ins(%37) outs(%42) (UL : [%arg3, %arg4], LR : [%43, %arg4]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %45 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %44 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %46, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d1i0_d2i1_d0i2__f012__n_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %22 = arith.muli %arg3, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %20, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c4 : index
                  %44 = arith.addi %43, %c3 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%43, %arg3], LR : [%44, %arg3]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d1i0_d2i1_d0i2__f012__n_n_dim_x_level0_bc4__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %22 = arith.muli %arg3, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %20, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c4 : index
                  %44 = arith.addi %43, %c3 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%43, %arg3], LR : [%44, %arg3]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d1i0_d2i1_d0i2__f012__n_dim_y_level0_bc8_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %22 = arith.muli %arg3, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %20, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c4 : index
                  %44 = arith.addi %43, %c3 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%43, %arg3], LR : [%44, %arg3]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d1i0_d2i1_d0i2__f012__n_dim_y_level0_bc8_dim_x_level0_bc4__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %22 = arith.muli %arg3, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %20, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.muli %arg4, %c4 : index
                  %44 = arith.addi %43, %c3 : index
                  %45 = loom.reduce_sum ins(%37) outs(%42) (UL : [%43, %arg3], LR : [%44, %arg3]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %46 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%46], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %47 = loom.bufferize_to_memref %45 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %47, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012__n_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %arg4, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.addi %arg3, %c6 : index
                  %44 = loom.reduce_sum ins(%37) outs(%42) (UL : [%arg3, %arg4], LR : [%43, %arg4]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %45 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %44 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %46, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012__n_n_dim_x_level0_bc4__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %arg4, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.addi %arg3, %c6 : index
                  %44 = loom.reduce_sum ins(%37) outs(%42) (UL : [%arg3, %arg4], LR : [%43, %arg4]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %45 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %44 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %46, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012__dim_y_level0_bc8_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %arg4, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.addi %arg3, %c6 : index
                  %44 = loom.reduce_sum ins(%37) outs(%42) (UL : [%arg3, %arg4], LR : [%43, %arg4]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %45 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %44 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %46, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
                }
              } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @split_k_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012__dim_y_level0_bc8_n_dim_x_level0_bc4__tile_k512__tile_m32__tile_n32(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %22 = arith.muli %20, %c32 : index
                %23 = arith.muli %21, %c512 : index
                %24 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
                %25 = loom.semaphore_take %24 : memref<32x512xf16> -> memref<32x512xf16>
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [32, 512], strides: [4096, 1] : memref<256x4096xf16> to memref<32x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %27 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %28 = arith.muli %arg4, %c32 : index
                %29 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %30 = loom.semaphore_take %29 : memref<512x32xf16> -> memref<512x32xf16>
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %32 = loom.bufferize_to_tensor %30[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %33 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %34 = loom.semaphore_take %33 : memref<32x32xf16> -> memref<32x32xf16>
                %35 = loom.init_tensor %34[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %36 = linalg.fill ins(%cst : f16) outs(%35 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %37 = linalg.matmul ins(%27, %32 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%36 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %30 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %41 {
                  %42 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %43 = arith.addi %arg3, %c6 : index
                  %44 = loom.reduce_sum ins(%37) outs(%42) (UL : [%arg3, %arg4], LR : [%43, %arg4]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %34 : memref<32x32xf16>
                  %45 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %28)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %46 = loom.bufferize_to_memref %44 : tensor<32x32xf16> -> memref<32x32xf16>
                  loom.copy %46, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %39 : memref<32x32xf16>
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
