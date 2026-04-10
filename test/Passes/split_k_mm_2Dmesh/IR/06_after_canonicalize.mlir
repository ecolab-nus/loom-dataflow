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
                %27 = arith.muli %arg4, %c2 : index
                %28 = arith.addi %arg3, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %21, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg4, %c2 : index
                %35 = arith.addi %arg3, %34 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %35], LR : [%arg5, %35]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %36 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %38 = loom.semaphore_take %37 : memref<32x32xf16> -> memref<32x32xf16>
                %39 = loom.init_tensor %38[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %41 = linalg.matmul ins(%29, %36 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %42 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %43 = loom.semaphore_take %42 : memref<32x32xf16> -> memref<32x32xf16>
                %44 = loom.init_tensor %43[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %45 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %45 {
                  %46 = linalg.fill ins(%cst : f16) outs(%44 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %47 = arith.muli %arg4, %c2 : index
                  %48 = arith.addi %arg3, %47 : index
                  %49 = loom.reduce_sum ins(%41) outs(%46) region : (UL : [%c0, %48], LR : [%c7, %48]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %38 : memref<32x32xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<32x32xf16> -> memref<32x32xf16>
                  %52 = arith.muli %arg4, %c2 : index
                  %53 = arith.addi %arg3, %52 : index
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %53], LR : [%arg5, %53]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<32x32xf16>
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
                %27 = arith.muli %arg4, %c2 : index
                %28 = arith.addi %arg3, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %21, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg4, %c2 : index
                %35 = arith.muli %arg4, %c2 : index
                %36 = arith.addi %35, %c1 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %34], LR : [%arg5, %36]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %37 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %42 = linalg.matmul ins(%29, %37 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%41 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %43 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %44 = loom.semaphore_take %43 : memref<32x32xf16> -> memref<32x32xf16>
                %45 = loom.init_tensor %44[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %46 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %46 {
                  %47 = linalg.fill ins(%cst : f16) outs(%45 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %48 = arith.muli %arg4, %c2 : index
                  %49 = arith.addi %arg3, %48 : index
                  %50 = loom.reduce_sum ins(%42) outs(%47) region : (UL : [%c0, %49], LR : [%c7, %49]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %39 : memref<32x32xf16>
                  %51 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%51], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %52 = loom.bufferize_to_memref %50 : tensor<32x32xf16> -> memref<32x32xf16>
                  %53 = arith.muli %arg4, %c2 : index
                  %54 = arith.addi %arg3, %53 : index
                  loom.copy %52, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %54], LR : [%arg5, %54]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %44 : memref<32x32xf16>
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
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
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
                %27 = arith.muli %arg3, %c2 : index
                %28 = arith.addi %arg4, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %21, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg3, %c2 : index
                %35 = arith.addi %arg4, %34 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %35], LR : [%arg5, %35]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %36 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %38 = loom.semaphore_take %37 : memref<32x32xf16> -> memref<32x32xf16>
                %39 = loom.init_tensor %38[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %41 = linalg.matmul ins(%29, %36 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %42 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %43 = loom.semaphore_take %42 : memref<32x32xf16> -> memref<32x32xf16>
                %44 = loom.init_tensor %43[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %45 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %45 {
                  %46 = linalg.fill ins(%cst : f16) outs(%44 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %47 = arith.muli %arg4, %c4 : index
                  %48 = arith.addi %arg3, %47 : index
                  %49 = loom.reduce_sum ins(%41) outs(%46) region : (UL : [%c0, %48], LR : [%c7, %48]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %38 : memref<32x32xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<32x32xf16> -> memref<32x32xf16>
                  %52 = arith.muli %arg3, %c2 : index
                  %53 = arith.addi %arg4, %52 : index
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %53], LR : [%arg5, %53]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<32x32xf16>
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
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
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
                %27 = arith.muli %arg3, %c2 : index
                %28 = arith.muli %arg3, %c2 : index
                %29 = arith.addi %28, %c1 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %27], LR : [%arg5, %29]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %30 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %31 = arith.muli %21, %c32 : index
                %32 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %33 = loom.semaphore_take %32 : memref<512x32xf16> -> memref<512x32xf16>
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %31)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %35 = arith.muli %arg3, %c2 : index
                %36 = arith.addi %arg4, %35 : index
                loom.copy %reinterpret_cast_0, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %36], LR : [%arg5, %36]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %37 = loom.bufferize_to_tensor %33[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %42 = linalg.matmul ins(%30, %37 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%41 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %33 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %43 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %44 = loom.semaphore_take %43 : memref<32x32xf16> -> memref<32x32xf16>
                %45 = loom.init_tensor %44[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %46 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %46 {
                  %47 = linalg.fill ins(%cst : f16) outs(%45 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %48 = arith.muli %arg4, %c4 : index
                  %49 = arith.addi %arg3, %48 : index
                  %50 = loom.reduce_sum ins(%42) outs(%47) region : (UL : [%c0, %49], LR : [%c7, %49]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %39 : memref<32x32xf16>
                  %51 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %31)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%51], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %52 = loom.bufferize_to_memref %50 : tensor<32x32xf16> -> memref<32x32xf16>
                  %53 = arith.muli %arg3, %c2 : index
                  %54 = arith.addi %arg4, %53 : index
                  loom.copy %52, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %54], LR : [%arg5, %54]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %44 : memref<32x32xf16>
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
                %27 = arith.muli %arg4, %c4 : index
                %28 = arith.addi %arg3, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %21, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg4, %c4 : index
                %35 = arith.addi %arg3, %34 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %35], LR : [%arg5, %35]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %36 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %38 = loom.semaphore_take %37 : memref<32x32xf16> -> memref<32x32xf16>
                %39 = loom.init_tensor %38[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %41 = linalg.matmul ins(%29, %36 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %42 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %43 = loom.semaphore_take %42 : memref<32x32xf16> -> memref<32x32xf16>
                %44 = loom.init_tensor %43[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %45 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %45 {
                  %46 = linalg.fill ins(%cst : f16) outs(%44 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %47 = arith.muli %arg4, %c4 : index
                  %48 = arith.addi %arg3, %47 : index
                  %49 = loom.reduce_sum ins(%41) outs(%46) region : (UL : [%c0, %48], LR : [%c7, %48]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %38 : memref<32x32xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<32x32xf16> -> memref<32x32xf16>
                  %52 = arith.muli %arg4, %c4 : index
                  %53 = arith.addi %arg3, %52 : index
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %53], LR : [%arg5, %53]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<32x32xf16>
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
      %c3 = arith.constant 3 : index
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
                %27 = arith.muli %arg4, %c4 : index
                %28 = arith.addi %arg3, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %21, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg4, %c4 : index
                %35 = arith.muli %arg4, %c4 : index
                %36 = arith.addi %35, %c3 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %34], LR : [%arg5, %36]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %37 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %42 = linalg.matmul ins(%29, %37 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%41 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %43 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %44 = loom.semaphore_take %43 : memref<32x32xf16> -> memref<32x32xf16>
                %45 = loom.init_tensor %44[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %46 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %46 {
                  %47 = linalg.fill ins(%cst : f16) outs(%45 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %48 = arith.muli %arg4, %c4 : index
                  %49 = arith.addi %arg3, %48 : index
                  %50 = loom.reduce_sum ins(%42) outs(%47) region : (UL : [%c0, %49], LR : [%c7, %49]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %39 : memref<32x32xf16>
                  %51 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%51], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %52 = loom.bufferize_to_memref %50 : tensor<32x32xf16> -> memref<32x32xf16>
                  %53 = arith.muli %arg4, %c4 : index
                  %54 = arith.addi %arg3, %53 : index
                  loom.copy %52, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %54], LR : [%arg5, %54]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %44 : memref<32x32xf16>
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
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
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
                %27 = arith.muli %arg3, %c4 : index
                %28 = arith.addi %arg4, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %21, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg3, %c4 : index
                %35 = arith.addi %arg4, %34 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %35], LR : [%arg5, %35]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %36 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %38 = loom.semaphore_take %37 : memref<32x32xf16> -> memref<32x32xf16>
                %39 = loom.init_tensor %38[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %41 = linalg.matmul ins(%29, %36 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %42 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %43 = loom.semaphore_take %42 : memref<32x32xf16> -> memref<32x32xf16>
                %44 = loom.init_tensor %43[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %45 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %45 {
                  %46 = linalg.fill ins(%cst : f16) outs(%44 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %47 = arith.muli %arg4, %c2 : index
                  %48 = arith.addi %arg3, %47 : index
                  %49 = loom.reduce_sum ins(%41) outs(%46) region : (UL : [%c0, %48], LR : [%c7, %48]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %38 : memref<32x32xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<32x32xf16> -> memref<32x32xf16>
                  %52 = arith.muli %arg3, %c4 : index
                  %53 = arith.addi %arg4, %52 : index
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %53], LR : [%arg5, %53]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<32x32xf16>
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
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
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
                %27 = arith.muli %arg3, %c4 : index
                %28 = arith.muli %arg3, %c4 : index
                %29 = arith.addi %28, %c3 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %27], LR : [%arg5, %29]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %30 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %31 = arith.muli %21, %c32 : index
                %32 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %33 = loom.semaphore_take %32 : memref<512x32xf16> -> memref<512x32xf16>
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %31)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %35 = arith.muli %arg3, %c4 : index
                %36 = arith.addi %arg4, %35 : index
                loom.copy %reinterpret_cast_0, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %36], LR : [%arg5, %36]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %37 = loom.bufferize_to_tensor %33[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %38 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %39 = loom.semaphore_take %38 : memref<32x32xf16> -> memref<32x32xf16>
                %40 = loom.init_tensor %39[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %41 = linalg.fill ins(%cst : f16) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %42 = linalg.matmul ins(%30, %37 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%41 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %33 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %43 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %44 = loom.semaphore_take %43 : memref<32x32xf16> -> memref<32x32xf16>
                %45 = loom.init_tensor %44[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %46 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %46 {
                  %47 = linalg.fill ins(%cst : f16) outs(%45 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %48 = arith.muli %arg4, %c2 : index
                  %49 = arith.addi %arg3, %48 : index
                  %50 = loom.reduce_sum ins(%42) outs(%47) region : (UL : [%c0, %49], LR : [%c7, %49]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %39 : memref<32x32xf16>
                  %51 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %31)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%51], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %52 = loom.bufferize_to_memref %50 : tensor<32x32xf16> -> memref<32x32xf16>
                  %53 = arith.muli %arg3, %c4 : index
                  %54 = arith.addi %arg4, %53 : index
                  loom.copy %52, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %54], LR : [%arg5, %54]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %44 : memref<32x32xf16>
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
                %27 = arith.muli %arg4, %c2 : index
                %28 = arith.addi %arg5, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %arg3], LR : [%28, %arg3]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %20, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg4, %c2 : index
                %35 = arith.addi %arg5, %34 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %arg3], LR : [%35, %arg3]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %36 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %38 = loom.semaphore_take %37 : memref<32x32xf16> -> memref<32x32xf16>
                %39 = loom.init_tensor %38[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %41 = linalg.matmul ins(%29, %36 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %42 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %43 = loom.semaphore_take %42 : memref<32x32xf16> -> memref<32x32xf16>
                %44 = loom.init_tensor %43[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %45 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %45 {
                  %46 = linalg.fill ins(%cst : f16) outs(%44 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %47 = arith.muli %arg4, %c2 : index
                  %48 = arith.addi %47, %c1 : index
                  %49 = loom.reduce_sum ins(%41) outs(%46) region : (UL : [%47, %arg3], LR : [%48, %arg3]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %38 : memref<32x32xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<32x32xf16> -> memref<32x32xf16>
                  %52 = arith.muli %arg4, %c2 : index
                  %53 = arith.addi %arg5, %52 : index
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%53, %arg3], LR : [%53, %arg3]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<32x32xf16>
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
      %c7 = arith.constant 7 : index
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
                %27 = arith.muli %arg4, %c2 : index
                %28 = arith.addi %arg5, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %arg3], LR : [%28, %arg3]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %20, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg4, %c2 : index
                %35 = arith.addi %arg5, %34 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%35, %c0], LR : [%35, %c7]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %36 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %38 = loom.semaphore_take %37 : memref<32x32xf16> -> memref<32x32xf16>
                %39 = loom.init_tensor %38[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %41 = linalg.matmul ins(%29, %36 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %42 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %43 = loom.semaphore_take %42 : memref<32x32xf16> -> memref<32x32xf16>
                %44 = loom.init_tensor %43[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %45 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %45 {
                  %46 = linalg.fill ins(%cst : f16) outs(%44 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %47 = arith.muli %arg4, %c2 : index
                  %48 = arith.addi %47, %c1 : index
                  %49 = loom.reduce_sum ins(%41) outs(%46) region : (UL : [%47, %arg3], LR : [%48, %arg3]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %38 : memref<32x32xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<32x32xf16> -> memref<32x32xf16>
                  %52 = arith.muli %arg4, %c2 : index
                  %53 = arith.addi %arg5, %52 : index
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%53, %arg3], LR : [%53, %arg3]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<32x32xf16>
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
                %27 = arith.muli %arg3, %c2 : index
                %28 = arith.addi %arg5, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %arg4], LR : [%28, %arg4]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %arg4, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg3, %c2 : index
                %35 = arith.addi %arg5, %34 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %arg4], LR : [%35, %arg4]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %36 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %38 = loom.semaphore_take %37 : memref<32x32xf16> -> memref<32x32xf16>
                %39 = loom.init_tensor %38[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %41 = linalg.matmul ins(%29, %36 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %42 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %43 = loom.semaphore_take %42 : memref<32x32xf16> -> memref<32x32xf16>
                %44 = loom.init_tensor %43[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %45 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %45 {
                  %46 = linalg.fill ins(%cst : f16) outs(%44 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %47 = arith.addi %arg3, %c4 : index
                  %48 = loom.reduce_sum ins(%41) outs(%46) region : (UL : [%arg3, %arg4], LR : [%47, %arg4]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %38 : memref<32x32xf16>
                  %49 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %50 = loom.bufferize_to_memref %48 : tensor<32x32xf16> -> memref<32x32xf16>
                  %51 = arith.muli %arg3, %c2 : index
                  %52 = arith.addi %arg5, %51 : index
                  loom.copy %50, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%52, %arg4], LR : [%52, %arg4]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<32x32xf16>
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
      %c7 = arith.constant 7 : index
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
                %27 = arith.muli %arg3, %c2 : index
                %28 = arith.addi %arg5, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%28, %c0], LR : [%28, %c7]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %arg4, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg3, %c2 : index
                %35 = arith.addi %arg5, %34 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %arg4], LR : [%35, %arg4]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %36 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %38 = loom.semaphore_take %37 : memref<32x32xf16> -> memref<32x32xf16>
                %39 = loom.init_tensor %38[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %41 = linalg.matmul ins(%29, %36 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %42 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %43 = loom.semaphore_take %42 : memref<32x32xf16> -> memref<32x32xf16>
                %44 = loom.init_tensor %43[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %45 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %45 {
                  %46 = linalg.fill ins(%cst : f16) outs(%44 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %47 = arith.addi %arg3, %c4 : index
                  %48 = loom.reduce_sum ins(%41) outs(%46) region : (UL : [%arg3, %arg4], LR : [%47, %arg4]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %38 : memref<32x32xf16>
                  %49 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %50 = loom.bufferize_to_memref %48 : tensor<32x32xf16> -> memref<32x32xf16>
                  %51 = arith.muli %arg3, %c2 : index
                  %52 = arith.addi %arg5, %51 : index
                  loom.copy %50, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%52, %arg4], LR : [%52, %arg4]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<32x32xf16>
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
                %27 = arith.muli %arg4, %c4 : index
                %28 = arith.addi %arg5, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %arg3], LR : [%28, %arg3]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %20, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg4, %c4 : index
                %35 = arith.addi %arg5, %34 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %arg3], LR : [%35, %arg3]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %36 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %38 = loom.semaphore_take %37 : memref<32x32xf16> -> memref<32x32xf16>
                %39 = loom.init_tensor %38[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %41 = linalg.matmul ins(%29, %36 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %42 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %43 = loom.semaphore_take %42 : memref<32x32xf16> -> memref<32x32xf16>
                %44 = loom.init_tensor %43[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %45 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %45 {
                  %46 = linalg.fill ins(%cst : f16) outs(%44 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %47 = arith.muli %arg4, %c4 : index
                  %48 = arith.addi %47, %c3 : index
                  %49 = loom.reduce_sum ins(%41) outs(%46) region : (UL : [%47, %arg3], LR : [%48, %arg3]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %38 : memref<32x32xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<32x32xf16> -> memref<32x32xf16>
                  %52 = arith.muli %arg4, %c4 : index
                  %53 = arith.addi %arg5, %52 : index
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%53, %arg3], LR : [%53, %arg3]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<32x32xf16>
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
      %c7 = arith.constant 7 : index
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
                %27 = arith.muli %arg4, %c4 : index
                %28 = arith.addi %arg5, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %arg3], LR : [%28, %arg3]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %20, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg4, %c4 : index
                %35 = arith.addi %arg5, %34 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%35, %c0], LR : [%35, %c7]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %36 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %38 = loom.semaphore_take %37 : memref<32x32xf16> -> memref<32x32xf16>
                %39 = loom.init_tensor %38[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %41 = linalg.matmul ins(%29, %36 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %42 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %43 = loom.semaphore_take %42 : memref<32x32xf16> -> memref<32x32xf16>
                %44 = loom.init_tensor %43[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %45 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %45 {
                  %46 = linalg.fill ins(%cst : f16) outs(%44 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %47 = arith.muli %arg4, %c4 : index
                  %48 = arith.addi %47, %c3 : index
                  %49 = loom.reduce_sum ins(%41) outs(%46) region : (UL : [%47, %arg3], LR : [%48, %arg3]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %38 : memref<32x32xf16>
                  %50 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%50], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %51 = loom.bufferize_to_memref %49 : tensor<32x32xf16> -> memref<32x32xf16>
                  %52 = arith.muli %arg4, %c4 : index
                  %53 = arith.addi %arg5, %52 : index
                  loom.copy %51, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%53, %arg3], LR : [%53, %arg3]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<32x32xf16>
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
      %c6 = arith.constant 6 : index
      %c4 = arith.constant 4 : index
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
                %27 = arith.muli %arg3, %c4 : index
                %28 = arith.addi %arg5, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %arg4], LR : [%28, %arg4]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %arg4, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg3, %c4 : index
                %35 = arith.addi %arg5, %34 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %arg4], LR : [%35, %arg4]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %36 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %38 = loom.semaphore_take %37 : memref<32x32xf16> -> memref<32x32xf16>
                %39 = loom.init_tensor %38[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %41 = linalg.matmul ins(%29, %36 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %42 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %43 = loom.semaphore_take %42 : memref<32x32xf16> -> memref<32x32xf16>
                %44 = loom.init_tensor %43[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %45 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %45 {
                  %46 = linalg.fill ins(%cst : f16) outs(%44 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %47 = arith.addi %arg3, %c6 : index
                  %48 = loom.reduce_sum ins(%41) outs(%46) region : (UL : [%arg3, %arg4], LR : [%47, %arg4]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %38 : memref<32x32xf16>
                  %49 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %50 = loom.bufferize_to_memref %48 : tensor<32x32xf16> -> memref<32x32xf16>
                  %51 = arith.muli %arg3, %c4 : index
                  %52 = arith.addi %arg5, %51 : index
                  loom.copy %50, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%52, %arg4], LR : [%52, %arg4]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<32x32xf16>
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
      %c7 = arith.constant 7 : index
      %c6 = arith.constant 6 : index
      %c4 = arith.constant 4 : index
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
                %27 = arith.muli %arg3, %c4 : index
                %28 = arith.addi %arg5, %27 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%28, %c0], LR : [%28, %c7]) : memref<32x512xf16, strided<[4096, 1], offset: ?>> to memref<32x512xf16>
                %29 = loom.bufferize_to_tensor %25[32, 512] : memref<32x512xf16> -> tensor<32x512xf16>
                %30 = arith.muli %arg4, %c32 : index
                %31 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
                %32 = loom.semaphore_take %31 : memref<512x32xf16> -> memref<512x32xf16>
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%23, %30)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [512, 32], strides: [256, 1] : memref<4096x256xf16> to memref<512x32xf16, strided<[256, 1], offset: ?>>
                %34 = arith.muli %arg3, %c4 : index
                %35 = arith.addi %arg5, %34 : index
                loom.copy %reinterpret_cast_0, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %arg4], LR : [%35, %arg4]) : memref<512x32xf16, strided<[256, 1], offset: ?>> to memref<512x32xf16>
                %36 = loom.bufferize_to_tensor %32[512, 32] : memref<512x32xf16> -> tensor<512x32xf16>
                %37 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %38 = loom.semaphore_take %37 : memref<32x32xf16> -> memref<32x32xf16>
                %39 = loom.init_tensor %38[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %40 = linalg.fill ins(%cst : f16) outs(%39 : tensor<32x32xf16>) -> tensor<32x32xf16>
                %41 = linalg.matmul ins(%29, %36 : tensor<32x512xf16>, tensor<512x32xf16>) outs(%40 : tensor<32x32xf16>) -> tensor<32x32xf16>
                loom.semaphore_give %32 : memref<512x32xf16>
                loom.semaphore_give %25 : memref<32x512xf16>
                %42 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
                %43 = loom.semaphore_take %42 : memref<32x32xf16> -> memref<32x32xf16>
                %44 = loom.init_tensor %43[32, 32] : memref<32x32xf16> -> tensor<32x32xf16>
                %45 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %45 {
                  %46 = linalg.fill ins(%cst : f16) outs(%44 : tensor<32x32xf16>) -> tensor<32x32xf16>
                  %47 = arith.addi %arg3, %c6 : index
                  %48 = loom.reduce_sum ins(%41) outs(%46) region : (UL : [%arg3, %arg4], LR : [%47, %arg4]) : tensor<32x32xf16> -> tensor<32x32xf16>
                  loom.semaphore_give %38 : memref<32x32xf16>
                  %49 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%22, %30)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [32, 32], strides: [256, 1] : memref<256x256xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  %50 = loom.bufferize_to_memref %48 : tensor<32x32xf16> -> memref<32x32xf16>
                  %51 = arith.muli %arg3, %c4 : index
                  %52 = arith.addi %arg5, %51 : index
                  loom.copy %50, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%52, %arg4], LR : [%52, %arg4]) : memref<32x32xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
                  loom.semaphore_give %43 : memref<32x32xf16>
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
