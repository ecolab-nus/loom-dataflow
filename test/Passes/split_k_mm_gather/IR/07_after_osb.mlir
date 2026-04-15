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
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = arith.muli %21, %c262144 : index
            %28 = arith.addi %27, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %29 = arith.muli %arg4, %c2 : index
            %30 = arith.addi %arg3, %29 : index
            loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %30], LR : [%arg5, %30]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            %31 = arith.muli %23, %c64 : index
            %32 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %33 = loom.semaphore_take %32 : memref<512x64xf16> -> memref<512x64xf16>
            %34 = arith.muli %arg5, %c262144 : index
            %35 = arith.addi %34, %31 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
            %36 = arith.addi %29, %c1 : index
            loom.copy %reinterpret_cast_0, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %29], LR : [%arg5, %36]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
            %37 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %38 = loom.semaphore_take %37 : memref<64x64xf16> -> memref<64x64xf16>
            %39 = loom.semaphore_take %37 : memref<64x64xf16> -> memref<64x64xf16>
            linalg.fill ins(%cst : f16) outs(%39 : memref<64x64xf16>)
            linalg.matmul ins(%26, %33 : memref<64x512xf16>, memref<512x64xf16>) outs(%39 : memref<64x64xf16>)
            loom.semaphore_give %33 : memref<512x64xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            %40 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %40 {
              %41 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
              %42 = loom.semaphore_take %41 : memref<8x64x64xf16> -> memref<8x64x64xf16>
              loom.gather ins(%39 : memref<64x64xf16>) outs(%42 : memref<8x64x64xf16>) across(%arg5 : index) region : (UL : [%c0, %30], LR : [%c7, %30])
              loom.semaphore_give %39 : memref<64x64xf16>
              linalg.fill ins(%cst : f16) outs(%38 : memref<64x64xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%42 : memref<8x64x64xf16>) outs(%38 : memref<64x64xf16>) {
              ^bb0(%in: f16, %out: f16):
                %45 = arith.addf %in, %out : f16
                linalg.yield %45 : f16
              }
              loom.semaphore_give %42 : memref<8x64x64xf16>
              %43 = arith.muli %21, %c32768 : index
              %44 = arith.addi %43, %31 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %38, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %30], LR : [%arg5, %30]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %38 : memref<64x64xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 1, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y2y4__d0i2_d1i1_d2i0__f012__dim_y_level0_bc2_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = arith.muli %21, %c262144 : index
            %28 = arith.addi %27, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %29 = arith.muli %arg3, %c2 : index
            %30 = arith.addi %29, %c1 : index
            loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%arg5, %29], LR : [%arg5, %30]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            %31 = arith.muli %23, %c64 : index
            %32 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %33 = loom.semaphore_take %32 : memref<512x64xf16> -> memref<512x64xf16>
            %34 = arith.muli %arg5, %c262144 : index
            %35 = arith.addi %34, %31 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
            %36 = arith.addi %arg4, %29 : index
            loom.copy %reinterpret_cast_0, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %36], LR : [%arg5, %36]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
            %37 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %38 = loom.semaphore_take %37 : memref<64x64xf16> -> memref<64x64xf16>
            %39 = loom.semaphore_take %37 : memref<64x64xf16> -> memref<64x64xf16>
            linalg.fill ins(%cst : f16) outs(%39 : memref<64x64xf16>)
            linalg.matmul ins(%26, %33 : memref<64x512xf16>, memref<512x64xf16>) outs(%39 : memref<64x64xf16>)
            loom.semaphore_give %33 : memref<512x64xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            %40 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %40 {
              %41 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
              %42 = loom.semaphore_take %41 : memref<8x64x64xf16> -> memref<8x64x64xf16>
              loom.gather ins(%39 : memref<64x64xf16>) outs(%42 : memref<8x64x64xf16>) across(%arg5 : index) region : (UL : [%c0, %36], LR : [%c7, %36])
              loom.semaphore_give %39 : memref<64x64xf16>
              linalg.fill ins(%cst : f16) outs(%38 : memref<64x64xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%42 : memref<8x64x64xf16>) outs(%38 : memref<64x64xf16>) {
              ^bb0(%in: f16, %out: f16):
                %45 = arith.addf %in, %out : f16
                linalg.yield %45 : f16
              }
              loom.semaphore_give %42 : memref<8x64x64xf16>
              %43 = arith.muli %21, %c32768 : index
              %44 = arith.addi %43, %31 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %38, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %36], LR : [%arg5, %36]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %38 : memref<64x64xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc4_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = arith.muli %21, %c262144 : index
            %28 = arith.addi %27, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %29 = arith.muli %arg4, %c4 : index
            %30 = arith.addi %arg3, %29 : index
            loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %30], LR : [%arg5, %30]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            %31 = arith.muli %23, %c64 : index
            %32 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %33 = loom.semaphore_take %32 : memref<512x64xf16> -> memref<512x64xf16>
            %34 = arith.muli %arg5, %c262144 : index
            %35 = arith.addi %34, %31 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
            %36 = arith.addi %29, %c3 : index
            loom.copy %reinterpret_cast_0, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %29], LR : [%arg5, %36]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
            %37 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %38 = loom.semaphore_take %37 : memref<64x64xf16> -> memref<64x64xf16>
            %39 = loom.semaphore_take %37 : memref<64x64xf16> -> memref<64x64xf16>
            linalg.fill ins(%cst : f16) outs(%39 : memref<64x64xf16>)
            linalg.matmul ins(%26, %33 : memref<64x512xf16>, memref<512x64xf16>) outs(%39 : memref<64x64xf16>)
            loom.semaphore_give %33 : memref<512x64xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            %40 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %40 {
              %41 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
              %42 = loom.semaphore_take %41 : memref<8x64x64xf16> -> memref<8x64x64xf16>
              loom.gather ins(%39 : memref<64x64xf16>) outs(%42 : memref<8x64x64xf16>) across(%arg5 : index) region : (UL : [%c0, %30], LR : [%c7, %30])
              loom.semaphore_give %39 : memref<64x64xf16>
              linalg.fill ins(%cst : f16) outs(%38 : memref<64x64xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%42 : memref<8x64x64xf16>) outs(%38 : memref<64x64xf16>) {
              ^bb0(%in: f16, %out: f16):
                %45 = arith.addf %in, %out : f16
                linalg.yield %45 : f16
              }
              loom.semaphore_give %42 : memref<8x64x64xf16>
              %43 = arith.muli %21, %c32768 : index
              %44 = arith.addi %43, %31 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %38, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %30], LR : [%arg5, %30]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %38 : memref<64x64xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 1, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x8_y4y2__d0i2_d1i1_d2i0__f012__dim_y_level0_bc4_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = arith.muli %arg5, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = arith.muli %21, %c262144 : index
            %28 = arith.addi %27, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %29 = arith.muli %arg3, %c4 : index
            %30 = arith.addi %29, %c3 : index
            loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%arg5, %29], LR : [%arg5, %30]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            %31 = arith.muli %23, %c64 : index
            %32 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %33 = loom.semaphore_take %32 : memref<512x64xf16> -> memref<512x64xf16>
            %34 = arith.muli %arg5, %c262144 : index
            %35 = arith.addi %34, %31 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
            %36 = arith.addi %arg4, %29 : index
            loom.copy %reinterpret_cast_0, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %36], LR : [%arg5, %36]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
            %37 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %38 = loom.semaphore_take %37 : memref<64x64xf16> -> memref<64x64xf16>
            %39 = loom.semaphore_take %37 : memref<64x64xf16> -> memref<64x64xf16>
            linalg.fill ins(%cst : f16) outs(%39 : memref<64x64xf16>)
            linalg.matmul ins(%26, %33 : memref<64x512xf16>, memref<512x64xf16>) outs(%39 : memref<64x64xf16>)
            loom.semaphore_give %33 : memref<512x64xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            %40 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %40 {
              %41 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
              %42 = loom.semaphore_take %41 : memref<8x64x64xf16> -> memref<8x64x64xf16>
              loom.gather ins(%39 : memref<64x64xf16>) outs(%42 : memref<8x64x64xf16>) across(%arg5 : index) region : (UL : [%c0, %36], LR : [%c7, %36])
              loom.semaphore_give %39 : memref<64x64xf16>
              linalg.fill ins(%cst : f16) outs(%38 : memref<64x64xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%42 : memref<8x64x64xf16>) outs(%38 : memref<64x64xf16>) {
              ^bb0(%in: f16, %out: f16):
                %45 = arith.addf %in, %out : f16
                linalg.yield %45 : f16
              }
              loom.semaphore_give %42 : memref<8x64x64xf16>
              %43 = arith.muli %21, %c32768 : index
              %44 = arith.addi %43, %31 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %38, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %36], LR : [%arg5, %36]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %38 : memref<64x64xf16>
            }
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c8, %c4, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = arith.muli %23, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = arith.muli %arg3, %c262144 : index
            %28 = arith.addi %27, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %29 = arith.muli %arg4, %c2 : index
            %30 = arith.addi %arg5, %29 : index
            loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %arg3], LR : [%30, %arg3]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            %31 = arith.muli %21, %c64 : index
            %32 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %33 = loom.semaphore_take %32 : memref<512x64xf16> -> memref<512x64xf16>
            %34 = arith.muli %23, %c262144 : index
            %35 = arith.addi %34, %31 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%30, %c0], LR : [%30, %c7]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
            %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
            %38 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
            linalg.fill ins(%cst : f16) outs(%38 : memref<64x64xf16>)
            linalg.matmul ins(%26, %33 : memref<64x512xf16>, memref<512x64xf16>) outs(%38 : memref<64x64xf16>)
            loom.semaphore_give %33 : memref<512x64xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            %39 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %39 {
              %40 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
              %41 = loom.semaphore_take %40 : memref<8x64x64xf16> -> memref<8x64x64xf16>
              %42 = arith.addi %29, %c1 : index
              loom.gather ins(%38 : memref<64x64xf16>) outs(%41 : memref<8x64x64xf16>) across(%arg5 : index) region : (UL : [%29, %arg3], LR : [%42, %arg3])
              loom.semaphore_give %38 : memref<64x64xf16>
              linalg.fill ins(%cst : f16) outs(%37 : memref<64x64xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%41 : memref<8x64x64xf16>) outs(%37 : memref<64x64xf16>) {
              ^bb0(%in: f16, %out: f16):
                %45 = arith.addf %in, %out : f16
                linalg.yield %45 : f16
              }
              loom.semaphore_give %41 : memref<8x64x64xf16>
              %43 = arith.muli %arg3, %c32768 : index
              %44 = arith.addi %43, %31 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %37, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%30, %arg3], LR : [%30, %arg3]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %37 : memref<64x64xf16>
            }
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 1, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x2x4_y8__d0i2_d1i1_d2i0__f012__dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c262144 = arith.constant 262144 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = arith.muli %23, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = arith.muli %21, %c262144 : index
            %28 = arith.addi %27, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %29 = arith.muli %arg3, %c2 : index
            %30 = arith.addi %arg5, %29 : index
            loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%30, %c0], LR : [%30, %c7]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            %31 = arith.muli %arg4, %c64 : index
            %32 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %33 = loom.semaphore_take %32 : memref<512x64xf16> -> memref<512x64xf16>
            %34 = arith.muli %23, %c262144 : index
            %35 = arith.addi %34, %31 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %arg4], LR : [%30, %arg4]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
            %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
            %38 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
            linalg.fill ins(%cst : f16) outs(%38 : memref<64x64xf16>)
            linalg.matmul ins(%26, %33 : memref<64x512xf16>, memref<512x64xf16>) outs(%38 : memref<64x64xf16>)
            loom.semaphore_give %33 : memref<512x64xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            %39 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %39 {
              %40 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
              %41 = loom.semaphore_take %40 : memref<8x64x64xf16> -> memref<8x64x64xf16>
              %42 = arith.addi %29, %c1 : index
              loom.gather ins(%38 : memref<64x64xf16>) outs(%41 : memref<8x64x64xf16>) across(%arg5 : index) region : (UL : [%29, %arg4], LR : [%42, %arg4])
              loom.semaphore_give %38 : memref<64x64xf16>
              linalg.fill ins(%cst : f16) outs(%37 : memref<64x64xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%41 : memref<8x64x64xf16>) outs(%37 : memref<64x64xf16>) {
              ^bb0(%in: f16, %out: f16):
                %45 = arith.addf %in, %out : f16
                linalg.yield %45 : f16
              }
              loom.semaphore_give %41 : memref<8x64x64xf16>
              %43 = arith.muli %21, %c32768 : index
              %44 = arith.addi %43, %31 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %37, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%30, %arg4], LR : [%30, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %37 : memref<64x64xf16>
            }
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i0_d2i1__f012__n_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
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
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c8, %c2, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = arith.muli %23, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = arith.muli %arg3, %c262144 : index
            %28 = arith.addi %27, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %29 = arith.muli %arg4, %c4 : index
            %30 = arith.addi %arg5, %29 : index
            loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %arg3], LR : [%30, %arg3]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            %31 = arith.muli %21, %c64 : index
            %32 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %33 = loom.semaphore_take %32 : memref<512x64xf16> -> memref<512x64xf16>
            %34 = arith.muli %23, %c262144 : index
            %35 = arith.addi %34, %31 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%30, %c0], LR : [%30, %c7]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
            %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
            %38 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
            linalg.fill ins(%cst : f16) outs(%38 : memref<64x64xf16>)
            linalg.matmul ins(%26, %33 : memref<64x512xf16>, memref<512x64xf16>) outs(%38 : memref<64x64xf16>)
            loom.semaphore_give %33 : memref<512x64xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            %39 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %39 {
              %40 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
              %41 = loom.semaphore_take %40 : memref<8x64x64xf16> -> memref<8x64x64xf16>
              %42 = arith.addi %29, %c3 : index
              loom.gather ins(%38 : memref<64x64xf16>) outs(%41 : memref<8x64x64xf16>) across(%arg5 : index) region : (UL : [%29, %arg3], LR : [%42, %arg3])
              loom.semaphore_give %38 : memref<64x64xf16>
              linalg.fill ins(%cst : f16) outs(%37 : memref<64x64xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%41 : memref<8x64x64xf16>) outs(%37 : memref<64x64xf16>) {
              ^bb0(%in: f16, %out: f16):
                %45 = arith.addf %in, %out : f16
                linalg.yield %45 : f16
              }
              loom.semaphore_give %41 : memref<8x64x64xf16>
              %43 = arith.muli %arg3, %c32768 : index
              %44 = arith.addi %43, %31 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %37, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%30, %arg3], LR : [%30, %arg3]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %37 : memref<64x64xf16>
            }
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 1, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
    func.func @split_k_matmul_gather__x4x2_y8__d0i2_d1i1_d2i0__f012__dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
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
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = arith.muli %23, %c512 : index
            %25 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %26 = loom.semaphore_take %25 : memref<64x512xf16> -> memref<64x512xf16>
            %27 = arith.muli %21, %c262144 : index
            %28 = arith.addi %27, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            %29 = arith.muli %arg3, %c4 : index
            %30 = arith.addi %arg5, %29 : index
            loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%30, %c0], LR : [%30, %c7]) : memref<64x512xf16, strided<[4096, 1], offset: ?>> to memref<64x512xf16>
            %31 = arith.muli %arg4, %c64 : index
            %32 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %33 = loom.semaphore_take %32 : memref<512x64xf16> -> memref<512x64xf16>
            %34 = arith.muli %23, %c262144 : index
            %35 = arith.addi %34, %31 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %arg4], LR : [%30, %arg4]) : memref<512x64xf16, strided<[512, 1], offset: ?>> to memref<512x64xf16>
            %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
            %38 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
            linalg.fill ins(%cst : f16) outs(%38 : memref<64x64xf16>)
            linalg.matmul ins(%26, %33 : memref<64x512xf16>, memref<512x64xf16>) outs(%38 : memref<64x64xf16>)
            loom.semaphore_give %33 : memref<512x64xf16>
            loom.semaphore_give %26 : memref<64x512xf16>
            %39 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %39 {
              %40 = loom.alloc [8, 64, 64] on @L1 : memref<8x64x64xf16>
              %41 = loom.semaphore_take %40 : memref<8x64x64xf16> -> memref<8x64x64xf16>
              %42 = arith.addi %29, %c3 : index
              loom.gather ins(%38 : memref<64x64xf16>) outs(%41 : memref<8x64x64xf16>) across(%arg5 : index) region : (UL : [%29, %arg4], LR : [%42, %arg4])
              loom.semaphore_give %38 : memref<64x64xf16>
              linalg.fill ins(%cst : f16) outs(%37 : memref<64x64xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%41 : memref<8x64x64xf16>) outs(%37 : memref<64x64xf16>) {
              ^bb0(%in: f16, %out: f16):
                %45 = arith.addf %in, %out : f16
                linalg.yield %45 : f16
              }
              loom.semaphore_give %41 : memref<8x64x64xf16>
              %43 = arith.muli %21, %c32768 : index
              %44 = arith.addi %43, %31 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [64, 64], strides: [512, 1] : memref<512x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %37, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%30, %arg4], LR : [%30, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.semaphore_give %37 : memref<64x64xf16>
            }
          } {loom.block_sym = @tile_k, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
}
