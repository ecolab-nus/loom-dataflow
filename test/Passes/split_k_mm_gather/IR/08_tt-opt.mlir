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
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = arith.muli %21, %c262144 : index
            %29 = arith.addi %28, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %30 = arith.muli %arg4, %c2 : index
            %31 = arith.addi %arg3, %30 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %31], LR : [%arg5, %31]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%27 : memref<64x512xf16>) outs(%26 : memref<64x512xf16>)
            loom.semaphore_give %27 : memref<64x512xf16>
            %32 = arith.muli %23, %c32 : index
            %33 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %34 = loom.semaphore_take %33 : memref<512x32xf16> -> memref<512x32xf16>
            %35 = loom.semaphore_take %33 : memref<512x32xf16> -> memref<512x32xf16>
            %36 = arith.muli %arg5, %c262144 : index
            %37 = arith.addi %36, %32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            %38 = arith.addi %30, %c1 : index
            loom.copy %reinterpret_cast_0, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %30], LR : [%arg5, %38]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%35 : memref<512x32xf16>) outs(%34 : memref<512x32xf16>)
            loom.semaphore_give %35 : memref<512x32xf16>
            %39 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %40 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            %41 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            %42 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%26, %34 : memref<64x512xf16>, memref<512x32xf16>) outs(%43 : memref<64x32xf16>)
            loom.semaphore_give %34 : memref<512x32xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            loom.sync ins(%43 : memref<64x32xf16>) outs(%42 : memref<64x32xf16>)
            loom.semaphore_give %43 : memref<64x32xf16>
            %44 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %45 = loom.semaphore_take %44 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %46 = loom.semaphore_take %44 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            loom.gather ins(%42 : memref<64x32xf16>) outs(%46 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%c0, %31], LR : [%c7, %31])
            loom.semaphore_give %42 : memref<64x32xf16>
            %47 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %47 {
              linalg.fill ins(%cst : f16) outs(%41 : memref<64x32xf16>)
              loom.sync ins(%46 : memref<8x64x32xf16>) outs(%45 : memref<8x64x32xf16>)
              loom.semaphore_give %46 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%45 : memref<8x64x32xf16>) outs(%41 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %50 = arith.addf %in, %out : f16
                linalg.yield %50 : f16
              }
              loom.semaphore_give %45 : memref<8x64x32xf16>
              loom.sync ins(%41 : memref<64x32xf16>) outs(%40 : memref<64x32xf16>)
              loom.semaphore_give %41 : memref<64x32xf16>
              %48 = arith.muli %21, %c32768 : index
              %49 = arith.addi %48, %32 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%49], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %40, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %31], LR : [%arg5, %31]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %40 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 1, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y2y4__d0i2_d1i1_d2i0__f012__dim_y_level0_bc2_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = arith.muli %21, %c262144 : index
            %29 = arith.addi %28, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %30 = arith.muli %arg3, %c2 : index
            %31 = arith.addi %30, %c1 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %30], LR : [%arg5, %31]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%27 : memref<64x512xf16>) outs(%26 : memref<64x512xf16>)
            loom.semaphore_give %27 : memref<64x512xf16>
            %32 = arith.muli %23, %c32 : index
            %33 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %34 = loom.semaphore_take %33 : memref<512x32xf16> -> memref<512x32xf16>
            %35 = loom.semaphore_take %33 : memref<512x32xf16> -> memref<512x32xf16>
            %36 = arith.muli %arg5, %c262144 : index
            %37 = arith.addi %36, %32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            %38 = arith.addi %arg4, %30 : index
            loom.copy %reinterpret_cast_0, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%35 : memref<512x32xf16>) outs(%34 : memref<512x32xf16>)
            loom.semaphore_give %35 : memref<512x32xf16>
            %39 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %40 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            %41 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            %42 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%26, %34 : memref<64x512xf16>, memref<512x32xf16>) outs(%43 : memref<64x32xf16>)
            loom.semaphore_give %34 : memref<512x32xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            loom.sync ins(%43 : memref<64x32xf16>) outs(%42 : memref<64x32xf16>)
            loom.semaphore_give %43 : memref<64x32xf16>
            %44 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %45 = loom.semaphore_take %44 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %46 = loom.semaphore_take %44 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            loom.gather ins(%42 : memref<64x32xf16>) outs(%46 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%c0, %38], LR : [%c7, %38])
            loom.semaphore_give %42 : memref<64x32xf16>
            %47 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %47 {
              linalg.fill ins(%cst : f16) outs(%41 : memref<64x32xf16>)
              loom.sync ins(%46 : memref<8x64x32xf16>) outs(%45 : memref<8x64x32xf16>)
              loom.semaphore_give %46 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%45 : memref<8x64x32xf16>) outs(%41 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %50 = arith.addf %in, %out : f16
                linalg.yield %50 : f16
              }
              loom.semaphore_give %45 : memref<8x64x32xf16>
              loom.sync ins(%41 : memref<64x32xf16>) outs(%40 : memref<64x32xf16>)
              loom.semaphore_give %41 : memref<64x32xf16>
              %48 = arith.muli %21, %c32768 : index
              %49 = arith.addi %48, %32 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%49], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %40, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %40 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc4_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = arith.muli %21, %c262144 : index
            %29 = arith.addi %28, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %30 = arith.muli %arg4, %c4 : index
            %31 = arith.addi %arg3, %30 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %31], LR : [%arg5, %31]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%27 : memref<64x512xf16>) outs(%26 : memref<64x512xf16>)
            loom.semaphore_give %27 : memref<64x512xf16>
            %32 = arith.muli %23, %c32 : index
            %33 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %34 = loom.semaphore_take %33 : memref<512x32xf16> -> memref<512x32xf16>
            %35 = loom.semaphore_take %33 : memref<512x32xf16> -> memref<512x32xf16>
            %36 = arith.muli %arg5, %c262144 : index
            %37 = arith.addi %36, %32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            %38 = arith.addi %30, %c3 : index
            loom.copy %reinterpret_cast_0, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %30], LR : [%arg5, %38]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%35 : memref<512x32xf16>) outs(%34 : memref<512x32xf16>)
            loom.semaphore_give %35 : memref<512x32xf16>
            %39 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %40 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            %41 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            %42 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%26, %34 : memref<64x512xf16>, memref<512x32xf16>) outs(%43 : memref<64x32xf16>)
            loom.semaphore_give %34 : memref<512x32xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            loom.sync ins(%43 : memref<64x32xf16>) outs(%42 : memref<64x32xf16>)
            loom.semaphore_give %43 : memref<64x32xf16>
            %44 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %45 = loom.semaphore_take %44 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %46 = loom.semaphore_take %44 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            loom.gather ins(%42 : memref<64x32xf16>) outs(%46 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%c0, %31], LR : [%c7, %31])
            loom.semaphore_give %42 : memref<64x32xf16>
            %47 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %47 {
              linalg.fill ins(%cst : f16) outs(%41 : memref<64x32xf16>)
              loom.sync ins(%46 : memref<8x64x32xf16>) outs(%45 : memref<8x64x32xf16>)
              loom.semaphore_give %46 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%45 : memref<8x64x32xf16>) outs(%41 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %50 = arith.addf %in, %out : f16
                linalg.yield %50 : f16
              }
              loom.semaphore_give %45 : memref<8x64x32xf16>
              loom.sync ins(%41 : memref<64x32xf16>) outs(%40 : memref<64x32xf16>)
              loom.semaphore_give %41 : memref<64x32xf16>
              %48 = arith.muli %21, %c32768 : index
              %49 = arith.addi %48, %32 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%49], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %40, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %31], LR : [%arg5, %31]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %40 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 1, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i1_d2i0__f012__dim_y_level0_bc4_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = arith.muli %21, %c262144 : index
            %29 = arith.addi %28, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %30 = arith.muli %arg3, %c4 : index
            %31 = arith.addi %30, %c3 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %30], LR : [%arg5, %31]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%27 : memref<64x512xf16>) outs(%26 : memref<64x512xf16>)
            loom.semaphore_give %27 : memref<64x512xf16>
            %32 = arith.muli %23, %c32 : index
            %33 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %34 = loom.semaphore_take %33 : memref<512x32xf16> -> memref<512x32xf16>
            %35 = loom.semaphore_take %33 : memref<512x32xf16> -> memref<512x32xf16>
            %36 = arith.muli %arg5, %c262144 : index
            %37 = arith.addi %36, %32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            %38 = arith.addi %arg4, %30 : index
            loom.copy %reinterpret_cast_0, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%35 : memref<512x32xf16>) outs(%34 : memref<512x32xf16>)
            loom.semaphore_give %35 : memref<512x32xf16>
            %39 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %40 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            %41 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            %42 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%26, %34 : memref<64x512xf16>, memref<512x32xf16>) outs(%43 : memref<64x32xf16>)
            loom.semaphore_give %34 : memref<512x32xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            loom.sync ins(%43 : memref<64x32xf16>) outs(%42 : memref<64x32xf16>)
            loom.semaphore_give %43 : memref<64x32xf16>
            %44 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %45 = loom.semaphore_take %44 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %46 = loom.semaphore_take %44 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            loom.gather ins(%42 : memref<64x32xf16>) outs(%46 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%c0, %38], LR : [%c7, %38])
            loom.semaphore_give %42 : memref<64x32xf16>
            %47 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %47 {
              linalg.fill ins(%cst : f16) outs(%41 : memref<64x32xf16>)
              loom.sync ins(%46 : memref<8x64x32xf16>) outs(%45 : memref<8x64x32xf16>)
              loom.semaphore_give %46 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%45 : memref<8x64x32xf16>) outs(%41 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %50 = arith.addf %in, %out : f16
                linalg.yield %50 : f16
              }
              loom.semaphore_give %45 : memref<8x64x32xf16>
              loom.sync ins(%41 : memref<64x32xf16>) outs(%40 : memref<64x32xf16>)
              loom.semaphore_give %41 : memref<64x32xf16>
              %48 = arith.muli %21, %c32768 : index
              %49 = arith.addi %48, %32 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%49], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %40, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %38], LR : [%arg5, %38]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %40 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c8, %c4, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = arith.muli %23, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = arith.muli %arg3, %c262144 : index
            %29 = arith.addi %28, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %30 = arith.muli %arg4, %c2 : index
            %31 = arith.addi %arg5, %30 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %arg3], LR : [%31, %arg3]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%27 : memref<64x512xf16>) outs(%26 : memref<64x512xf16>)
            loom.semaphore_give %27 : memref<64x512xf16>
            %32 = arith.muli %21, %c32 : index
            %33 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %34 = loom.semaphore_take %33 : memref<512x32xf16> -> memref<512x32xf16>
            %35 = loom.semaphore_take %33 : memref<512x32xf16> -> memref<512x32xf16>
            %36 = arith.muli %23, %c262144 : index
            %37 = arith.addi %36, %32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%31, %c0], LR : [%31, %c7]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%35 : memref<512x32xf16>) outs(%34 : memref<512x32xf16>)
            loom.semaphore_give %35 : memref<512x32xf16>
            %38 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %39 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
            %40 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
            %41 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
            %42 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%26, %34 : memref<64x512xf16>, memref<512x32xf16>) outs(%42 : memref<64x32xf16>)
            loom.semaphore_give %34 : memref<512x32xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            loom.sync ins(%42 : memref<64x32xf16>) outs(%41 : memref<64x32xf16>)
            loom.semaphore_give %42 : memref<64x32xf16>
            %43 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %44 = loom.semaphore_take %43 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %45 = loom.semaphore_take %43 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %46 = arith.addi %30, %c1 : index
            loom.gather ins(%41 : memref<64x32xf16>) outs(%45 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%30, %arg3], LR : [%46, %arg3])
            loom.semaphore_give %41 : memref<64x32xf16>
            %47 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %47 {
              linalg.fill ins(%cst : f16) outs(%40 : memref<64x32xf16>)
              loom.sync ins(%45 : memref<8x64x32xf16>) outs(%44 : memref<8x64x32xf16>)
              loom.semaphore_give %45 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%44 : memref<8x64x32xf16>) outs(%40 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %50 = arith.addf %in, %out : f16
                linalg.yield %50 : f16
              }
              loom.semaphore_give %44 : memref<8x64x32xf16>
              loom.sync ins(%40 : memref<64x32xf16>) outs(%39 : memref<64x32xf16>)
              loom.semaphore_give %40 : memref<64x32xf16>
              %48 = arith.muli %arg3, %c32768 : index
              %49 = arith.addi %48, %32 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%49], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %39, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %arg3], LR : [%31, %arg3]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %39 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 1, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i1_d2i0__f012__dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          scf.for %arg7 = %c0 to %c2 step %c1 {
            scf.for %arg8 = %c0 to %c4 step %c1 {
              %20 = arith.muli %arg6, %c4 overflow<nsw> : index
              %21 = arith.addi %arg3, %20 : index
              %22 = arith.muli %arg7, %c8 overflow<nsw> : index
              %23 = arith.addi %arg4, %22 : index
              %24 = arith.muli %arg8, %c2 overflow<nsw> : index
              %25 = arith.addi %arg5, %24 : index
              %26 = arith.muli %25, %c512 : index
              %27 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
              %28 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
              %29 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
              %30 = arith.muli %21, %c262144 : index
              %31 = arith.addi %30, %26 : index
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
              %32 = arith.muli %arg3, %c2 : index
              %33 = arith.addi %arg5, %32 : index
              loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%33, %c0], LR : [%33, %c7]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
              loom.sync ins(%29 : memref<64x512xf16>) outs(%28 : memref<64x512xf16>)
              loom.semaphore_give %29 : memref<64x512xf16>
              %34 = arith.muli %23, %c32 : index
              %35 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
              %36 = loom.semaphore_take %35 : memref<512x32xf16> -> memref<512x32xf16>
              %37 = loom.semaphore_take %35 : memref<512x32xf16> -> memref<512x32xf16>
              %38 = arith.muli %25, %c262144 : index
              %39 = arith.addi %38, %34 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%39], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %arg4], LR : [%33, %arg4]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
              loom.sync ins(%37 : memref<512x32xf16>) outs(%36 : memref<512x32xf16>)
              loom.semaphore_give %37 : memref<512x32xf16>
              %40 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
              %41 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
              %42 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
              %43 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
              %44 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
              loom.matmul ins(%28, %36 : memref<64x512xf16>, memref<512x32xf16>) outs(%44 : memref<64x32xf16>)
              loom.semaphore_give %36 : memref<512x32xf16>
              loom.semaphore_give %28 : memref<64x512xf16>
              loom.sync ins(%44 : memref<64x32xf16>) outs(%43 : memref<64x32xf16>)
              loom.semaphore_give %44 : memref<64x32xf16>
              %45 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
              %46 = loom.semaphore_take %45 : memref<8x64x32xf16> -> memref<8x64x32xf16>
              %47 = loom.semaphore_take %45 : memref<8x64x32xf16> -> memref<8x64x32xf16>
              %48 = arith.addi %32, %c1 : index
              loom.gather ins(%43 : memref<64x32xf16>) outs(%47 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%32, %arg4], LR : [%48, %arg4])
              loom.semaphore_give %43 : memref<64x32xf16>
              %49 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %49 {
                linalg.fill ins(%cst : f16) outs(%42 : memref<64x32xf16>)
                loom.sync ins(%47 : memref<8x64x32xf16>) outs(%46 : memref<8x64x32xf16>)
                loom.semaphore_give %47 : memref<8x64x32xf16>
                linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%46 : memref<8x64x32xf16>) outs(%42 : memref<64x32xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %52 = arith.addf %in, %out : f16
                  linalg.yield %52 : f16
                }
                loom.semaphore_give %46 : memref<8x64x32xf16>
                loom.sync ins(%42 : memref<64x32xf16>) outs(%41 : memref<64x32xf16>)
                loom.semaphore_give %42 : memref<64x32xf16>
                %50 = arith.muli %21, %c32768 : index
                %51 = arith.addi %50, %34 : index
                %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%51], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                loom.copy %41, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %arg4], LR : [%33, %arg4]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                loom.semaphore_give %41 : memref<64x32xf16>
              }
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c8, %c2, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = arith.muli %23, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = arith.muli %arg3, %c262144 : index
            %29 = arith.addi %28, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %30 = arith.muli %arg4, %c4 : index
            %31 = arith.addi %arg5, %30 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %arg3], LR : [%31, %arg3]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%27 : memref<64x512xf16>) outs(%26 : memref<64x512xf16>)
            loom.semaphore_give %27 : memref<64x512xf16>
            %32 = arith.muli %21, %c32 : index
            %33 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %34 = loom.semaphore_take %33 : memref<512x32xf16> -> memref<512x32xf16>
            %35 = loom.semaphore_take %33 : memref<512x32xf16> -> memref<512x32xf16>
            %36 = arith.muli %23, %c262144 : index
            %37 = arith.addi %36, %32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%31, %c0], LR : [%31, %c7]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%35 : memref<512x32xf16>) outs(%34 : memref<512x32xf16>)
            loom.semaphore_give %35 : memref<512x32xf16>
            %38 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %39 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
            %40 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
            %41 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
            %42 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%26, %34 : memref<64x512xf16>, memref<512x32xf16>) outs(%42 : memref<64x32xf16>)
            loom.semaphore_give %34 : memref<512x32xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            loom.sync ins(%42 : memref<64x32xf16>) outs(%41 : memref<64x32xf16>)
            loom.semaphore_give %42 : memref<64x32xf16>
            %43 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %44 = loom.semaphore_take %43 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %45 = loom.semaphore_take %43 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %46 = arith.addi %30, %c3 : index
            loom.gather ins(%41 : memref<64x32xf16>) outs(%45 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%30, %arg3], LR : [%46, %arg3])
            loom.semaphore_give %41 : memref<64x32xf16>
            %47 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %47 {
              linalg.fill ins(%cst : f16) outs(%40 : memref<64x32xf16>)
              loom.sync ins(%45 : memref<8x64x32xf16>) outs(%44 : memref<8x64x32xf16>)
              loom.semaphore_give %45 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%44 : memref<8x64x32xf16>) outs(%40 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %50 = arith.addf %in, %out : f16
                linalg.yield %50 : f16
              }
              loom.semaphore_give %44 : memref<8x64x32xf16>
              loom.sync ins(%40 : memref<64x32xf16>) outs(%39 : memref<64x32xf16>)
              loom.semaphore_give %40 : memref<64x32xf16>
              %48 = arith.muli %arg3, %c32768 : index
              %49 = arith.addi %48, %32 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%49], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %39, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %arg3], LR : [%31, %arg3]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %39 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 1, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i1_d2i0__f012__dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c2 step %c1 {
            scf.for %arg8 = %c0 to %c2 step %c1 {
              %20 = arith.muli %arg6, %c2 overflow<nsw> : index
              %21 = arith.addi %arg3, %20 : index
              %22 = arith.muli %arg7, %c8 overflow<nsw> : index
              %23 = arith.addi %arg4, %22 : index
              %24 = arith.muli %arg8, %c4 overflow<nsw> : index
              %25 = arith.addi %arg5, %24 : index
              %26 = arith.muli %25, %c512 : index
              %27 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
              %28 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
              %29 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
              %30 = arith.muli %21, %c262144 : index
              %31 = arith.addi %30, %26 : index
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
              %32 = arith.muli %arg3, %c4 : index
              %33 = arith.addi %arg5, %32 : index
              loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%33, %c0], LR : [%33, %c7]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
              loom.sync ins(%29 : memref<64x512xf16>) outs(%28 : memref<64x512xf16>)
              loom.semaphore_give %29 : memref<64x512xf16>
              %34 = arith.muli %23, %c32 : index
              %35 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
              %36 = loom.semaphore_take %35 : memref<512x32xf16> -> memref<512x32xf16>
              %37 = loom.semaphore_take %35 : memref<512x32xf16> -> memref<512x32xf16>
              %38 = arith.muli %25, %c262144 : index
              %39 = arith.addi %38, %34 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%39], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %arg4], LR : [%33, %arg4]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
              loom.sync ins(%37 : memref<512x32xf16>) outs(%36 : memref<512x32xf16>)
              loom.semaphore_give %37 : memref<512x32xf16>
              %40 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
              %41 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
              %42 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
              %43 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
              %44 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
              loom.matmul ins(%28, %36 : memref<64x512xf16>, memref<512x32xf16>) outs(%44 : memref<64x32xf16>)
              loom.semaphore_give %36 : memref<512x32xf16>
              loom.semaphore_give %28 : memref<64x512xf16>
              loom.sync ins(%44 : memref<64x32xf16>) outs(%43 : memref<64x32xf16>)
              loom.semaphore_give %44 : memref<64x32xf16>
              %45 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
              %46 = loom.semaphore_take %45 : memref<8x64x32xf16> -> memref<8x64x32xf16>
              %47 = loom.semaphore_take %45 : memref<8x64x32xf16> -> memref<8x64x32xf16>
              %48 = arith.addi %32, %c3 : index
              loom.gather ins(%43 : memref<64x32xf16>) outs(%47 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%32, %arg4], LR : [%48, %arg4])
              loom.semaphore_give %43 : memref<64x32xf16>
              %49 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %49 {
                linalg.fill ins(%cst : f16) outs(%42 : memref<64x32xf16>)
                loom.sync ins(%47 : memref<8x64x32xf16>) outs(%46 : memref<8x64x32xf16>)
                loom.semaphore_give %47 : memref<8x64x32xf16>
                linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%46 : memref<8x64x32xf16>) outs(%42 : memref<64x32xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %52 = arith.addf %in, %out : f16
                  linalg.yield %52 : f16
                }
                loom.semaphore_give %46 : memref<8x64x32xf16>
                loom.sync ins(%42 : memref<64x32xf16>) outs(%41 : memref<64x32xf16>)
                loom.semaphore_give %42 : memref<64x32xf16>
                %50 = arith.muli %21, %c32768 : index
                %51 = arith.addi %50, %34 : index
                %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%51], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                loom.copy %41, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %arg4], LR : [%33, %arg4]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                loom.semaphore_give %41 : memref<64x32xf16>
              }
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y2y2y2__d0i2_d1i2_d2i0_d3i1__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%c2, %c2, %c8, %c2) step (%c1, %c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg8, %c2 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c8 overflow<nsw> : index
            %25 = arith.addi %24, %arg6 : index
            %26 = arith.muli %25, %c512 : index
            %27 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %28 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %29 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %30 = arith.muli %21, %c262144 : index
            %31 = arith.addi %30, %26 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %32 = arith.muli %arg3, %c2 : index
            %33 = arith.addi %arg6, %32 : index
            %34 = arith.muli %arg4, %c4 : index
            %35 = arith.addi %33, %34 : index
            loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %35], LR : [%arg5, %35]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%29 : memref<64x512xf16>) outs(%28 : memref<64x512xf16>)
            loom.semaphore_give %29 : memref<64x512xf16>
            %36 = arith.muli %23, %c32 : index
            %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %39 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %40 = arith.muli %25, %c262144 : index
            %41 = arith.addi %40, %36 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %35], LR : [%arg5, %35]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%39 : memref<512x32xf16>) outs(%38 : memref<512x32xf16>)
            loom.semaphore_give %39 : memref<512x32xf16>
            %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %46 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%28, %38 : memref<64x512xf16>, memref<512x32xf16>) outs(%46 : memref<64x32xf16>)
            loom.semaphore_give %38 : memref<512x32xf16>
            loom.semaphore_give %28 : memref<64x512xf16>
            loom.sync ins(%46 : memref<64x32xf16>) outs(%45 : memref<64x32xf16>)
            loom.semaphore_give %46 : memref<64x32xf16>
            %47 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %48 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %49 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            loom.gather ins(%45 : memref<64x32xf16>) outs(%49 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%c0, %35], LR : [%c7, %35])
            loom.semaphore_give %45 : memref<64x32xf16>
            %50 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %50 {
              linalg.fill ins(%cst : f16) outs(%44 : memref<64x32xf16>)
              loom.sync ins(%49 : memref<8x64x32xf16>) outs(%48 : memref<8x64x32xf16>)
              loom.semaphore_give %49 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%48 : memref<8x64x32xf16>) outs(%44 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %53 = arith.addf %in, %out : f16
                linalg.yield %53 : f16
              }
              loom.semaphore_give %48 : memref<8x64x32xf16>
              loom.sync ins(%44 : memref<64x32xf16>) outs(%43 : memref<64x32xf16>)
              loom.semaphore_give %44 : memref<64x32xf16>
              %51 = arith.muli %21, %c32768 : index
              %52 = arith.addi %51, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%52], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %43, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %35], LR : [%arg5, %35]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %43 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 2, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y2y2y2__d0i2_d1i2_d2i1_d3i0__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%c2, %c2, %c8, %c2) step (%c1, %c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg8, %c2 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c8 overflow<nsw> : index
            %25 = arith.addi %24, %arg6 : index
            %26 = arith.muli %25, %c512 : index
            %27 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %28 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %29 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %30 = arith.muli %21, %c262144 : index
            %31 = arith.addi %30, %26 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %32 = arith.muli %arg4, %c2 : index
            %33 = arith.addi %arg6, %32 : index
            %34 = arith.muli %arg3, %c4 : index
            %35 = arith.addi %33, %34 : index
            loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %35], LR : [%arg5, %35]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%29 : memref<64x512xf16>) outs(%28 : memref<64x512xf16>)
            loom.semaphore_give %29 : memref<64x512xf16>
            %36 = arith.muli %23, %c32 : index
            %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %39 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %40 = arith.muli %25, %c262144 : index
            %41 = arith.addi %40, %36 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %35], LR : [%arg5, %35]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%39 : memref<512x32xf16>) outs(%38 : memref<512x32xf16>)
            loom.semaphore_give %39 : memref<512x32xf16>
            %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %46 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%28, %38 : memref<64x512xf16>, memref<512x32xf16>) outs(%46 : memref<64x32xf16>)
            loom.semaphore_give %38 : memref<512x32xf16>
            loom.semaphore_give %28 : memref<64x512xf16>
            loom.sync ins(%46 : memref<64x32xf16>) outs(%45 : memref<64x32xf16>)
            loom.semaphore_give %46 : memref<64x32xf16>
            %47 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %48 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %49 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            loom.gather ins(%45 : memref<64x32xf16>) outs(%49 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%c0, %35], LR : [%c7, %35])
            loom.semaphore_give %45 : memref<64x32xf16>
            %50 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %50 {
              linalg.fill ins(%cst : f16) outs(%44 : memref<64x32xf16>)
              loom.sync ins(%49 : memref<8x64x32xf16>) outs(%48 : memref<8x64x32xf16>)
              loom.semaphore_give %49 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%48 : memref<8x64x32xf16>) outs(%44 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %53 = arith.addf %in, %out : f16
                linalg.yield %53 : f16
              }
              loom.semaphore_give %48 : memref<8x64x32xf16>
              loom.sync ins(%44 : memref<64x32xf16>) outs(%43 : memref<64x32xf16>)
              loom.semaphore_give %44 : memref<64x32xf16>
              %51 = arith.muli %21, %c32768 : index
              %52 = arith.addi %51, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%52], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %43, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %35], LR : [%arg5, %35]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %43 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y2y4__d0i2_d1i2_d2i0_d3i1__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%c4, %c4, %c2, %c2) step (%c1, %c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          scf.for %arg8 = %c0 to %c4 step %c1 {
            scf.for %arg9 = %c0 to %c2 step %c1 {
              %20 = arith.muli %arg7, %c4 overflow<nsw> : index
              %21 = arith.addi %arg3, %20 : index
              %22 = arith.muli %arg8, %c4 overflow<nsw> : index
              %23 = arith.addi %arg4, %22 : index
              %24 = arith.muli %arg5, %c2 overflow<nsw> : index
              %25 = arith.addi %24, %arg6 : index
              %26 = arith.muli %arg9, %c4 overflow<nsw> : index
              %27 = arith.addi %25, %26 : index
              %28 = arith.muli %27, %c512 : index
              %29 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
              %30 = loom.semaphore_take %29 : memref<64x512xf16> -> memref<64x512xf16>
              %31 = loom.semaphore_take %29 : memref<64x512xf16> -> memref<64x512xf16>
              %32 = arith.muli %21, %c262144 : index
              %33 = arith.addi %32, %28 : index
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%33], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
              %34 = arith.muli %arg3, %c2 : index
              %35 = arith.addi %arg5, %34 : index
              %36 = arith.muli %arg4, %c2 : index
              %37 = arith.addi %arg6, %36 : index
              loom.copy %reinterpret_cast, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %37], LR : [%35, %37]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
              loom.sync ins(%31 : memref<64x512xf16>) outs(%30 : memref<64x512xf16>)
              loom.semaphore_give %31 : memref<64x512xf16>
              %38 = arith.muli %23, %c32 : index
              %39 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
              %40 = loom.semaphore_take %39 : memref<512x32xf16> -> memref<512x32xf16>
              %41 = loom.semaphore_take %39 : memref<512x32xf16> -> memref<512x32xf16>
              %42 = arith.muli %27, %c262144 : index
              %43 = arith.addi %42, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %37], LR : [%35, %37]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
              loom.sync ins(%41 : memref<512x32xf16>) outs(%40 : memref<512x32xf16>)
              loom.semaphore_give %41 : memref<512x32xf16>
              %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
              %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
              %46 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
              %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
              %48 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
              loom.matmul ins(%30, %40 : memref<64x512xf16>, memref<512x32xf16>) outs(%48 : memref<64x32xf16>)
              loom.semaphore_give %40 : memref<512x32xf16>
              loom.semaphore_give %30 : memref<64x512xf16>
              loom.sync ins(%48 : memref<64x32xf16>) outs(%47 : memref<64x32xf16>)
              loom.semaphore_give %48 : memref<64x32xf16>
              %49 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
              %50 = loom.semaphore_take %49 : memref<8x64x32xf16> -> memref<8x64x32xf16>
              %51 = loom.semaphore_take %49 : memref<8x64x32xf16> -> memref<8x64x32xf16>
              %52 = arith.addi %34, %c1 : index
              loom.gather ins(%47 : memref<64x32xf16>) outs(%51 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%34, %37], LR : [%52, %37])
              loom.semaphore_give %47 : memref<64x32xf16>
              %53 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %53 {
                linalg.fill ins(%cst : f16) outs(%46 : memref<64x32xf16>)
                loom.sync ins(%51 : memref<8x64x32xf16>) outs(%50 : memref<8x64x32xf16>)
                loom.semaphore_give %51 : memref<8x64x32xf16>
                linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%50 : memref<8x64x32xf16>) outs(%46 : memref<64x32xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %56 = arith.addf %in, %out : f16
                  linalg.yield %56 : f16
                }
                loom.semaphore_give %50 : memref<8x64x32xf16>
                loom.sync ins(%46 : memref<64x32xf16>) outs(%45 : memref<64x32xf16>)
                loom.semaphore_give %46 : memref<64x32xf16>
                %54 = arith.muli %21, %c32768 : index
                %55 = arith.addi %54, %38 : index
                %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%55], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                loom.copy %45, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%35, %37], LR : [%35, %37]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                loom.semaphore_give %45 : memref<64x32xf16>
              }
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y2y4__d0i2_d1i2_d2i1_d3i0__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%c4, %c4, %c2, %c2) step (%c1, %c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          scf.for %arg8 = %c0 to %c4 step %c1 {
            scf.for %arg9 = %c0 to %c2 step %c1 {
              %20 = arith.muli %arg7, %c4 overflow<nsw> : index
              %21 = arith.addi %arg3, %20 : index
              %22 = arith.muli %arg8, %c4 overflow<nsw> : index
              %23 = arith.addi %arg4, %22 : index
              %24 = arith.muli %arg5, %c2 overflow<nsw> : index
              %25 = arith.addi %24, %arg6 : index
              %26 = arith.muli %arg9, %c4 overflow<nsw> : index
              %27 = arith.addi %25, %26 : index
              %28 = arith.muli %27, %c512 : index
              %29 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
              %30 = loom.semaphore_take %29 : memref<64x512xf16> -> memref<64x512xf16>
              %31 = loom.semaphore_take %29 : memref<64x512xf16> -> memref<64x512xf16>
              %32 = arith.muli %21, %c262144 : index
              %33 = arith.addi %32, %28 : index
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%33], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
              %34 = arith.muli %arg4, %c2 : index
              %35 = arith.addi %arg5, %34 : index
              %36 = arith.muli %arg3, %c2 : index
              %37 = arith.addi %arg6, %36 : index
              loom.copy %reinterpret_cast, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %37], LR : [%35, %37]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
              loom.sync ins(%31 : memref<64x512xf16>) outs(%30 : memref<64x512xf16>)
              loom.semaphore_give %31 : memref<64x512xf16>
              %38 = arith.muli %23, %c32 : index
              %39 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
              %40 = loom.semaphore_take %39 : memref<512x32xf16> -> memref<512x32xf16>
              %41 = loom.semaphore_take %39 : memref<512x32xf16> -> memref<512x32xf16>
              %42 = arith.muli %27, %c262144 : index
              %43 = arith.addi %42, %38 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %37], LR : [%35, %37]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
              loom.sync ins(%41 : memref<512x32xf16>) outs(%40 : memref<512x32xf16>)
              loom.semaphore_give %41 : memref<512x32xf16>
              %44 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
              %45 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
              %46 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
              %47 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
              %48 = loom.semaphore_take %44 : memref<64x32xf16> -> memref<64x32xf16>
              loom.matmul ins(%30, %40 : memref<64x512xf16>, memref<512x32xf16>) outs(%48 : memref<64x32xf16>)
              loom.semaphore_give %40 : memref<512x32xf16>
              loom.semaphore_give %30 : memref<64x512xf16>
              loom.sync ins(%48 : memref<64x32xf16>) outs(%47 : memref<64x32xf16>)
              loom.semaphore_give %48 : memref<64x32xf16>
              %49 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
              %50 = loom.semaphore_take %49 : memref<8x64x32xf16> -> memref<8x64x32xf16>
              %51 = loom.semaphore_take %49 : memref<8x64x32xf16> -> memref<8x64x32xf16>
              %52 = arith.addi %34, %c1 : index
              loom.gather ins(%47 : memref<64x32xf16>) outs(%51 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%34, %37], LR : [%52, %37])
              loom.semaphore_give %47 : memref<64x32xf16>
              %53 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %53 {
                linalg.fill ins(%cst : f16) outs(%46 : memref<64x32xf16>)
                loom.sync ins(%51 : memref<8x64x32xf16>) outs(%50 : memref<8x64x32xf16>)
                loom.semaphore_give %51 : memref<8x64x32xf16>
                linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%50 : memref<8x64x32xf16>) outs(%46 : memref<64x32xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %56 = arith.addf %in, %out : f16
                  linalg.yield %56 : f16
                }
                loom.semaphore_give %50 : memref<8x64x32xf16>
                loom.sync ins(%46 : memref<64x32xf16>) outs(%45 : memref<64x32xf16>)
                loom.semaphore_give %46 : memref<64x32xf16>
                %54 = arith.muli %21, %c32768 : index
                %55 = arith.addi %54, %38 : index
                %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%55], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                loom.copy %45, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%35, %37], LR : [%35, %37]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
                loom.semaphore_give %45 : memref<64x32xf16>
              }
            } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y4y2__d0i2_d1i2_d2i0_d3i1__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%c4, %c2, %c2, %c4) step (%c1, %c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg8, %c2 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c2 overflow<nsw> : index
            %25 = arith.addi %24, %arg6 : index
            %26 = arith.muli %25, %c512 : index
            %27 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %28 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %29 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %30 = arith.muli %21, %c262144 : index
            %31 = arith.addi %30, %26 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %32 = arith.muli %arg3, %c2 : index
            %33 = arith.addi %arg5, %32 : index
            %34 = arith.muli %arg4, %c4 : index
            %35 = arith.addi %arg6, %34 : index
            loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%29 : memref<64x512xf16>) outs(%28 : memref<64x512xf16>)
            loom.semaphore_give %29 : memref<64x512xf16>
            %36 = arith.muli %23, %c32 : index
            %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %39 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %40 = arith.muli %25, %c262144 : index
            %41 = arith.addi %40, %36 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%39 : memref<512x32xf16>) outs(%38 : memref<512x32xf16>)
            loom.semaphore_give %39 : memref<512x32xf16>
            %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %46 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%28, %38 : memref<64x512xf16>, memref<512x32xf16>) outs(%46 : memref<64x32xf16>)
            loom.semaphore_give %38 : memref<512x32xf16>
            loom.semaphore_give %28 : memref<64x512xf16>
            loom.sync ins(%46 : memref<64x32xf16>) outs(%45 : memref<64x32xf16>)
            loom.semaphore_give %46 : memref<64x32xf16>
            %47 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %48 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %49 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %50 = arith.addi %32, %c1 : index
            loom.gather ins(%45 : memref<64x32xf16>) outs(%49 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%32, %35], LR : [%50, %35])
            loom.semaphore_give %45 : memref<64x32xf16>
            %51 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %51 {
              linalg.fill ins(%cst : f16) outs(%44 : memref<64x32xf16>)
              loom.sync ins(%49 : memref<8x64x32xf16>) outs(%48 : memref<8x64x32xf16>)
              loom.semaphore_give %49 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%48 : memref<8x64x32xf16>) outs(%44 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %54 = arith.addf %in, %out : f16
                linalg.yield %54 : f16
              }
              loom.semaphore_give %48 : memref<8x64x32xf16>
              loom.sync ins(%44 : memref<64x32xf16>) outs(%43 : memref<64x32xf16>)
              loom.semaphore_give %44 : memref<64x32xf16>
              %52 = arith.muli %21, %c32768 : index
              %53 = arith.addi %52, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%53], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %43, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %43 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y4y2__d0i2_d1i2_d2i1_d3i0__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%c2, %c4, %c2, %c4) step (%c1, %c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          scf.for %arg8 = %c0 to %c4 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg8, %c4 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c2 overflow<nsw> : index
            %25 = arith.addi %24, %arg6 : index
            %26 = arith.muli %25, %c512 : index
            %27 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %28 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %29 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %30 = arith.muli %21, %c262144 : index
            %31 = arith.addi %30, %26 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %32 = arith.muli %arg4, %c2 : index
            %33 = arith.addi %arg5, %32 : index
            %34 = arith.muli %arg3, %c4 : index
            %35 = arith.addi %arg6, %34 : index
            loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%29 : memref<64x512xf16>) outs(%28 : memref<64x512xf16>)
            loom.semaphore_give %29 : memref<64x512xf16>
            %36 = arith.muli %23, %c32 : index
            %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %39 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %40 = arith.muli %25, %c262144 : index
            %41 = arith.addi %40, %36 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%39 : memref<512x32xf16>) outs(%38 : memref<512x32xf16>)
            loom.semaphore_give %39 : memref<512x32xf16>
            %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %46 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%28, %38 : memref<64x512xf16>, memref<512x32xf16>) outs(%46 : memref<64x32xf16>)
            loom.semaphore_give %38 : memref<512x32xf16>
            loom.semaphore_give %28 : memref<64x512xf16>
            loom.sync ins(%46 : memref<64x32xf16>) outs(%45 : memref<64x32xf16>)
            loom.semaphore_give %46 : memref<64x32xf16>
            %47 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %48 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %49 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %50 = arith.addi %32, %c1 : index
            loom.gather ins(%45 : memref<64x32xf16>) outs(%49 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%32, %35], LR : [%50, %35])
            loom.semaphore_give %45 : memref<64x32xf16>
            %51 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %51 {
              linalg.fill ins(%cst : f16) outs(%44 : memref<64x32xf16>)
              loom.sync ins(%49 : memref<8x64x32xf16>) outs(%48 : memref<8x64x32xf16>)
              loom.semaphore_give %49 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%48 : memref<8x64x32xf16>) outs(%44 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %54 = arith.addf %in, %out : f16
                linalg.yield %54 : f16
              }
              loom.semaphore_give %48 : memref<8x64x32xf16>
              loom.sync ins(%44 : memref<64x32xf16>) outs(%43 : memref<64x32xf16>)
              loom.semaphore_give %44 : memref<64x32xf16>
              %52 = arith.muli %21, %c32768 : index
              %53 = arith.addi %52, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%53], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %43, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %43 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y2y4__d0i2_d1i2_d2i0_d3i1__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%c2, %c4, %c4, %c2) step (%c1, %c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          scf.for %arg8 = %c0 to %c4 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg8, %c4 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c4 overflow<nsw> : index
            %25 = arith.addi %24, %arg6 : index
            %26 = arith.muli %25, %c512 : index
            %27 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %28 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %29 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %30 = arith.muli %21, %c262144 : index
            %31 = arith.addi %30, %26 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %32 = arith.muli %arg3, %c4 : index
            %33 = arith.addi %arg5, %32 : index
            %34 = arith.muli %arg4, %c2 : index
            %35 = arith.addi %arg6, %34 : index
            loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%29 : memref<64x512xf16>) outs(%28 : memref<64x512xf16>)
            loom.semaphore_give %29 : memref<64x512xf16>
            %36 = arith.muli %23, %c32 : index
            %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %39 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %40 = arith.muli %25, %c262144 : index
            %41 = arith.addi %40, %36 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%39 : memref<512x32xf16>) outs(%38 : memref<512x32xf16>)
            loom.semaphore_give %39 : memref<512x32xf16>
            %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %46 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%28, %38 : memref<64x512xf16>, memref<512x32xf16>) outs(%46 : memref<64x32xf16>)
            loom.semaphore_give %38 : memref<512x32xf16>
            loom.semaphore_give %28 : memref<64x512xf16>
            loom.sync ins(%46 : memref<64x32xf16>) outs(%45 : memref<64x32xf16>)
            loom.semaphore_give %46 : memref<64x32xf16>
            %47 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %48 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %49 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %50 = arith.addi %32, %c3 : index
            loom.gather ins(%45 : memref<64x32xf16>) outs(%49 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%32, %35], LR : [%50, %35])
            loom.semaphore_give %45 : memref<64x32xf16>
            %51 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %51 {
              linalg.fill ins(%cst : f16) outs(%44 : memref<64x32xf16>)
              loom.sync ins(%49 : memref<8x64x32xf16>) outs(%48 : memref<8x64x32xf16>)
              loom.semaphore_give %49 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%48 : memref<8x64x32xf16>) outs(%44 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %54 = arith.addf %in, %out : f16
                linalg.yield %54 : f16
              }
              loom.semaphore_give %48 : memref<8x64x32xf16>
              loom.sync ins(%44 : memref<64x32xf16>) outs(%43 : memref<64x32xf16>)
              loom.semaphore_give %44 : memref<64x32xf16>
              %52 = arith.muli %21, %c32768 : index
              %53 = arith.addi %52, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%53], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %43, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %43 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y2y4__d0i2_d1i2_d2i1_d3i0__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c8 = arith.constant 8 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%c4, %c2, %c4, %c2) step (%c1, %c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg8, %c2 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c4 overflow<nsw> : index
            %25 = arith.addi %24, %arg6 : index
            %26 = arith.muli %25, %c512 : index
            %27 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %28 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %29 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %30 = arith.muli %21, %c262144 : index
            %31 = arith.addi %30, %26 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %32 = arith.muli %arg4, %c4 : index
            %33 = arith.addi %arg5, %32 : index
            %34 = arith.muli %arg3, %c2 : index
            %35 = arith.addi %arg6, %34 : index
            loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%29 : memref<64x512xf16>) outs(%28 : memref<64x512xf16>)
            loom.semaphore_give %29 : memref<64x512xf16>
            %36 = arith.muli %23, %c32 : index
            %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %39 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %40 = arith.muli %25, %c262144 : index
            %41 = arith.addi %40, %36 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%39 : memref<512x32xf16>) outs(%38 : memref<512x32xf16>)
            loom.semaphore_give %39 : memref<512x32xf16>
            %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %46 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%28, %38 : memref<64x512xf16>, memref<512x32xf16>) outs(%46 : memref<64x32xf16>)
            loom.semaphore_give %38 : memref<512x32xf16>
            loom.semaphore_give %28 : memref<64x512xf16>
            loom.sync ins(%46 : memref<64x32xf16>) outs(%45 : memref<64x32xf16>)
            loom.semaphore_give %46 : memref<64x32xf16>
            %47 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %48 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %49 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %50 = arith.addi %32, %c3 : index
            loom.gather ins(%45 : memref<64x32xf16>) outs(%49 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%32, %35], LR : [%50, %35])
            loom.semaphore_give %45 : memref<64x32xf16>
            %51 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %51 {
              linalg.fill ins(%cst : f16) outs(%44 : memref<64x32xf16>)
              loom.sync ins(%49 : memref<8x64x32xf16>) outs(%48 : memref<8x64x32xf16>)
              loom.semaphore_give %49 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%48 : memref<8x64x32xf16>) outs(%44 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %54 = arith.addf %in, %out : f16
                linalg.yield %54 : f16
              }
              loom.semaphore_give %48 : memref<8x64x32xf16>
              loom.sync ins(%44 : memref<64x32xf16>) outs(%43 : memref<64x32xf16>)
              loom.semaphore_give %44 : memref<64x32xf16>
              %52 = arith.muli %21, %c32768 : index
              %53 = arith.addi %52, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%53], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %43, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %43 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y4y2__d0i2_d1i2_d2i0_d3i1__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c8 = arith.constant 8 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%c2, %c2, %c4, %c4) step (%c1, %c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg8, %c2 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c4 overflow<nsw> : index
            %25 = arith.addi %24, %arg6 : index
            %26 = arith.muli %25, %c512 : index
            %27 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %28 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %29 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %30 = arith.muli %21, %c262144 : index
            %31 = arith.addi %30, %26 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %32 = arith.muli %arg3, %c4 : index
            %33 = arith.addi %arg5, %32 : index
            %34 = arith.muli %arg4, %c4 : index
            %35 = arith.addi %arg6, %34 : index
            loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%29 : memref<64x512xf16>) outs(%28 : memref<64x512xf16>)
            loom.semaphore_give %29 : memref<64x512xf16>
            %36 = arith.muli %23, %c32 : index
            %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %39 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %40 = arith.muli %25, %c262144 : index
            %41 = arith.addi %40, %36 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%39 : memref<512x32xf16>) outs(%38 : memref<512x32xf16>)
            loom.semaphore_give %39 : memref<512x32xf16>
            %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %46 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%28, %38 : memref<64x512xf16>, memref<512x32xf16>) outs(%46 : memref<64x32xf16>)
            loom.semaphore_give %38 : memref<512x32xf16>
            loom.semaphore_give %28 : memref<64x512xf16>
            loom.sync ins(%46 : memref<64x32xf16>) outs(%45 : memref<64x32xf16>)
            loom.semaphore_give %46 : memref<64x32xf16>
            %47 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %48 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %49 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %50 = arith.addi %32, %c3 : index
            loom.gather ins(%45 : memref<64x32xf16>) outs(%49 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%32, %35], LR : [%50, %35])
            loom.semaphore_give %45 : memref<64x32xf16>
            %51 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %51 {
              linalg.fill ins(%cst : f16) outs(%44 : memref<64x32xf16>)
              loom.sync ins(%49 : memref<8x64x32xf16>) outs(%48 : memref<8x64x32xf16>)
              loom.semaphore_give %49 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%48 : memref<8x64x32xf16>) outs(%44 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %54 = arith.addf %in, %out : f16
                linalg.yield %54 : f16
              }
              loom.semaphore_give %48 : memref<8x64x32xf16>
              loom.sync ins(%44 : memref<64x32xf16>) outs(%43 : memref<64x32xf16>)
              loom.semaphore_give %44 : memref<64x32xf16>
              %52 = arith.muli %21, %c32768 : index
              %53 = arith.addi %52, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%53], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %43, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %43 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y4y2__d0i2_d1i2_d2i1_d3i0__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c8 = arith.constant 8 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%c2, %c2, %c4, %c4) step (%c1, %c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg8, %c2 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c4 overflow<nsw> : index
            %25 = arith.addi %24, %arg6 : index
            %26 = arith.muli %25, %c512 : index
            %27 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %28 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %29 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %30 = arith.muli %21, %c262144 : index
            %31 = arith.addi %30, %26 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %32 = arith.muli %arg4, %c4 : index
            %33 = arith.addi %arg5, %32 : index
            %34 = arith.muli %arg3, %c4 : index
            %35 = arith.addi %arg6, %34 : index
            loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%29 : memref<64x512xf16>) outs(%28 : memref<64x512xf16>)
            loom.semaphore_give %29 : memref<64x512xf16>
            %36 = arith.muli %23, %c32 : index
            %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %39 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %40 = arith.muli %25, %c262144 : index
            %41 = arith.addi %40, %36 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%39 : memref<512x32xf16>) outs(%38 : memref<512x32xf16>)
            loom.semaphore_give %39 : memref<512x32xf16>
            %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %46 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%28, %38 : memref<64x512xf16>, memref<512x32xf16>) outs(%46 : memref<64x32xf16>)
            loom.semaphore_give %38 : memref<512x32xf16>
            loom.semaphore_give %28 : memref<64x512xf16>
            loom.sync ins(%46 : memref<64x32xf16>) outs(%45 : memref<64x32xf16>)
            loom.semaphore_give %46 : memref<64x32xf16>
            %47 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %48 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %49 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %50 = arith.addi %32, %c3 : index
            loom.gather ins(%45 : memref<64x32xf16>) outs(%49 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%32, %35], LR : [%50, %35])
            loom.semaphore_give %45 : memref<64x32xf16>
            %51 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %51 {
              linalg.fill ins(%cst : f16) outs(%44 : memref<64x32xf16>)
              loom.sync ins(%49 : memref<8x64x32xf16>) outs(%48 : memref<8x64x32xf16>)
              loom.semaphore_give %49 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%48 : memref<8x64x32xf16>) outs(%44 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %54 = arith.addf %in, %out : f16
                linalg.yield %54 : f16
              }
              loom.semaphore_give %48 : memref<8x64x32xf16>
              loom.sync ins(%44 : memref<64x32xf16>) outs(%43 : memref<64x32xf16>)
              loom.semaphore_give %44 : memref<64x32xf16>
              %52 = arith.muli %21, %c32768 : index
              %53 = arith.addi %52, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%53], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %43, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %35], LR : [%33, %35]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %43 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x2x2_y8__d0i2_d1i2_d2i0_d3i1__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c8) step (%c1, %c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg8, %c2 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c2 overflow<nsw> : index
            %25 = arith.addi %24, %arg6 : index
            %26 = arith.muli %25, %c512 : index
            %27 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %28 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %29 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %30 = arith.muli %21, %c262144 : index
            %31 = arith.addi %30, %26 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %32 = arith.muli %arg3, %c2 : index
            %33 = arith.addi %arg5, %32 : index
            %34 = arith.muli %arg4, %c4 : index
            %35 = arith.addi %33, %34 : index
            loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %arg6], LR : [%35, %arg6]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%29 : memref<64x512xf16>) outs(%28 : memref<64x512xf16>)
            loom.semaphore_give %29 : memref<64x512xf16>
            %36 = arith.muli %23, %c32 : index
            %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %39 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %40 = arith.muli %25, %c262144 : index
            %41 = arith.addi %40, %36 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %arg6], LR : [%35, %arg6]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%39 : memref<512x32xf16>) outs(%38 : memref<512x32xf16>)
            loom.semaphore_give %39 : memref<512x32xf16>
            %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %46 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%28, %38 : memref<64x512xf16>, memref<512x32xf16>) outs(%46 : memref<64x32xf16>)
            loom.semaphore_give %38 : memref<512x32xf16>
            loom.semaphore_give %28 : memref<64x512xf16>
            loom.sync ins(%46 : memref<64x32xf16>) outs(%45 : memref<64x32xf16>)
            loom.semaphore_give %46 : memref<64x32xf16>
            %47 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %48 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %49 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %50 = arith.addi %32, %34 : index
            %51 = arith.addi %32, %c1 : index
            %52 = arith.addi %51, %34 : index
            loom.gather ins(%45 : memref<64x32xf16>) outs(%49 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%50, %arg6], LR : [%52, %arg6])
            loom.semaphore_give %45 : memref<64x32xf16>
            %53 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %53 {
              linalg.fill ins(%cst : f16) outs(%44 : memref<64x32xf16>)
              loom.sync ins(%49 : memref<8x64x32xf16>) outs(%48 : memref<8x64x32xf16>)
              loom.semaphore_give %49 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%48 : memref<8x64x32xf16>) outs(%44 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %56 = arith.addf %in, %out : f16
                linalg.yield %56 : f16
              }
              loom.semaphore_give %48 : memref<8x64x32xf16>
              loom.sync ins(%44 : memref<64x32xf16>) outs(%43 : memref<64x32xf16>)
              loom.semaphore_give %44 : memref<64x32xf16>
              %54 = arith.muli %21, %c32768 : index
              %55 = arith.addi %54, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%55], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %43, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%35, %arg6], LR : [%35, %arg6]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %43 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 2, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x2x2_y8__d0i2_d1i2_d2i1_d3i0__f012__n_n_n__tile_k512__tile_m64__tile_n32(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c8) step (%c1, %c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg8, %c2 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c2 overflow<nsw> : index
            %25 = arith.addi %24, %arg6 : index
            %26 = arith.muli %25, %c512 : index
            %27 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %28 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %29 = loom.semaphore_take %27 : memref<64x512xf16> -> memref<64x512xf16>
            %30 = arith.muli %21, %c262144 : index
            %31 = arith.addi %30, %26 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %32 = arith.muli %arg4, %c2 : index
            %33 = arith.addi %arg5, %32 : index
            %34 = arith.muli %arg3, %c4 : index
            %35 = arith.addi %33, %34 : index
            loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %arg6], LR : [%35, %arg6]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            loom.sync ins(%29 : memref<64x512xf16>) outs(%28 : memref<64x512xf16>)
            loom.semaphore_give %29 : memref<64x512xf16>
            %36 = arith.muli %23, %c32 : index
            %37 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %38 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %39 = loom.semaphore_take %37 : memref<512x32xf16> -> memref<512x32xf16>
            %40 = arith.muli %25, %c262144 : index
            %41 = arith.addi %40, %36 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%41], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %arg6], LR : [%35, %arg6]) : memref<512x32xf16, strided<[512, 1], offset: ?>> to memref<512x32xf16>
            loom.sync ins(%39 : memref<512x32xf16>) outs(%38 : memref<512x32xf16>)
            loom.semaphore_give %39 : memref<512x32xf16>
            %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %45 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %46 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%28, %38 : memref<64x512xf16>, memref<512x32xf16>) outs(%46 : memref<64x32xf16>)
            loom.semaphore_give %38 : memref<512x32xf16>
            loom.semaphore_give %28 : memref<64x512xf16>
            loom.sync ins(%46 : memref<64x32xf16>) outs(%45 : memref<64x32xf16>)
            loom.semaphore_give %46 : memref<64x32xf16>
            %47 = loom.alloc [8, 64, 32] on @L1 : memref<8x64x32xf16>
            %48 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %49 = loom.semaphore_take %47 : memref<8x64x32xf16> -> memref<8x64x32xf16>
            %50 = arith.addi %32, %34 : index
            %51 = arith.addi %32, %c1 : index
            %52 = arith.addi %51, %34 : index
            loom.gather ins(%45 : memref<64x32xf16>) outs(%49 : memref<8x64x32xf16>) across(%arg5 : index) region : (UL : [%50, %arg6], LR : [%52, %arg6])
            loom.semaphore_give %45 : memref<64x32xf16>
            %53 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %53 {
              linalg.fill ins(%cst : f16) outs(%44 : memref<64x32xf16>)
              loom.sync ins(%49 : memref<8x64x32xf16>) outs(%48 : memref<8x64x32xf16>)
              loom.semaphore_give %49 : memref<8x64x32xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%48 : memref<8x64x32xf16>) outs(%44 : memref<64x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %56 = arith.addf %in, %out : f16
                linalg.yield %56 : f16
              }
              loom.semaphore_give %48 : memref<8x64x32xf16>
              loom.sync ins(%44 : memref<64x32xf16>) outs(%43 : memref<64x32xf16>)
              loom.semaphore_give %44 : memref<64x32xf16>
              %54 = arith.muli %21, %c32768 : index
              %55 = arith.addi %54, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%55], sizes: [64, 32], strides: [512, 1] : memref<512x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.copy %43, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%35, %arg6], LR : [%35, %arg6]) : memref<64x32xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %43 : memref<64x32xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_x, @dim_y]}
      return
    }
  }
}
