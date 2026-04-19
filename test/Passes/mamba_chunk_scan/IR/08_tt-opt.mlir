module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c16384 = arith.constant 16384 : index
      %c262144 = arith.constant 262144 : index
      %c2048 = arith.constant 2048 : index
      %c1048576 = arith.constant 1048576 : index
      %c1024 = arith.constant 1024 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 1.442380e+00 : f16
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c4, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg9, %20 : index
          %22 = arith.muli %arg8, %c32 : index
          %23 = arith.muli %21, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %27 = arith.muli %arg8, %c65536 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg11, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg12, %c2 : index
          %35 = arith.muli %arg8, %c4 : index
          %36 = arith.addi %34, %35 : index
          %37 = arith.addi %34, %c1 : index
          %38 = arith.addi %37, %35 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          %39 = loom.alloc [64] on @L1 : memref<64xf16>
          %40 = loom.semaphore_take %39 : memref<64xf16> -> memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25 : memref<64xf16>) outs(%40 : memref<64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %89 = arith.mulf %in, %cst_0 : f16
            %90 = math.powf %cst, %89 : f16
            linalg.yield %90 : f16
          }
          %41 = arith.divui %22, %c64 : index
          %42 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %43 = loom.semaphore_take %42 : memref<64x64xf16> -> memref<64x64xf16>
          %44 = arith.muli %arg12, %c65536 : index
          %45 = arith.addi %26, %44 : index
          %46 = arith.muli %41, %c64 overflow<nsw> : index
          %47 = arith.addi %45, %46 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          %48 = arith.addi %32, %c1 : index
          loom.copy %reinterpret_cast_1, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%48, %38]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          %49 = arith.muli %arg10, %c32 : index
          %50 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %51 = loom.semaphore_take %50 : memref<32x64xf16> -> memref<32x64xf16>
          %52 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %53 = arith.muli %arg12, %c1048576 : index
          %54 = arith.addi %52, %53 : index
          %55 = arith.muli %arg8, %c131072 : index
          %56 = arith.addi %54, %55 : index
          %57 = arith.muli %arg10, %c2048 : index
          %58 = arith.addi %56, %57 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%58], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          %59 = arith.addi %arg10, %34 : index
          %60 = arith.addi %59, %35 : index
          loom.copy %reinterpret_cast_2, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %60], LR : [%48, %60]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.transpose ins(%51 : memref<32x64xf16>) outs(%62 : memref<64x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %51 : memref<32x64xf16>
          %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          %65 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          loom.matmul ins(%43, %62 : memref<64x64xf16>, memref<64x32xf16>) outs(%64 : memref<64x32xf16>)
          loom.semaphore_give %62 : memref<64x32xf16>
          loom.semaphore_give %43 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %40 : memref<64x32xf16>, memref<64xf16>) outs(%65 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %89 = arith.mulf %in, %in_6 : f16
            linalg.yield %89 : f16
          }
          loom.semaphore_give %64 : memref<64x32xf16>
          loom.semaphore_give %40 : memref<64xf16>
          %66 = arith.addi %21, %c1 : index
          %67 = arith.muli %66, %c64 : index
          %68 = arith.ceildivui %67, %c64 : index
          %69 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %70 = loom.semaphore_take %69 : memref<64x64xf16> -> memref<64x64xf16>
          %71 = loom.alloc [64] on @L1 : memref<64xf16>
          %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
          %73 = loom.alloc [64] on @L1 : memref<64xf16>
          %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %75 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %76 = loom.semaphore_take %75 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %68 step %c1 {
            %89 = arith.muli %arg14, %c64 : index
            %90 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %91 = arith.muli %arg12, %c262144 : index
            %92 = arith.addi %90, %91 : index
            %93 = arith.muli %41, %c65536 overflow<nsw> : index
            %94 = arith.addi %92, %93 : index
            %95 = arith.muli %21, %c16384 : index
            %96 = arith.addi %94, %95 : index
            %97 = arith.addi %96, %89 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%97], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %98 = arith.addi %30, %89 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%98], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%48, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%98], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%48, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %25, %72, %74 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%70 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_10: f16, %in_11: f16, %in_12: f16, %out: f16):
              %105 = arith.mulf %in_11, %cst_0 : f16
              %106 = arith.mulf %in_10, %cst_0 : f16
              %107 = arith.subf %106, %105 : f16
              %108 = math.powf %cst, %107 : f16
              %109 = arith.mulf %in, %108 : f16
              %110 = arith.mulf %109, %in_12 : f16
              linalg.yield %110 : f16
            }
            loom.semaphore_give %74 : memref<64xf16>
            loom.semaphore_give %72 : memref<64xf16>
            %99 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %100 = arith.muli %arg12, %c4194304 : index
            %101 = arith.addi %99, %100 : index
            %102 = arith.muli %arg8, %c2048 : index
            %103 = arith.addi %101, %102 : index
            %104 = arith.addi %103, %49 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%104], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %60], LR : [%48, %60]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%70, %76 : memref<64x64xf16>, memref<64x32xf16>) outs(%65 : memref<64x32xf16>)
            loom.semaphore_give %76 : memref<64x32xf16>
            loom.semaphore_give %70 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %77 = loom.alloc [1] on @L1 : memref<f16>
          %78 = loom.semaphore_take %77 : memref<f16> -> memref<f16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %79 = arith.addi %35, %c3 : index
          loom.copy %reinterpret_cast_3, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %35], LR : [%c7, %79]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          %82 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %83 = arith.muli %arg12, %c4194304 : index
          %84 = arith.addi %82, %83 : index
          %85 = arith.muli %arg8, %c2048 : index
          %86 = arith.addi %84, %85 : index
          %87 = arith.addi %86, %49 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %60], LR : [%48, %60]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %81, %78 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%81 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
            %89 = arith.mulf %in_6, %in_7 : f16
            %90 = arith.addf %in, %89 : f16
            linalg.yield %90 : f16
          }
          loom.semaphore_give %78 : memref<f16>
          loom.semaphore_give %65 : memref<64x32xf16>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%87], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %88 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%81 : memref<64x32xf16>) outs(%88 : memref<64x32xf16>)
          loom.copy %88, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%32, %60], LR : [%48, %60]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %88 : memref<64x32xf16>
          loom.semaphore_give %81 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c16384 = arith.constant 16384 : index
      %c262144 = arith.constant 262144 : index
      %c2048 = arith.constant 2048 : index
      %c1048576 = arith.constant 1048576 : index
      %c1024 = arith.constant 1024 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 1.442380e+00 : f16
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c2, %c4) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg9, %20 : index
          %22 = arith.muli %arg8, %c32 : index
          %23 = arith.muli %21, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %27 = arith.muli %arg8, %c65536 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg12, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg11, %c2 : index
          %35 = arith.muli %arg8, %c4 : index
          %36 = arith.addi %34, %35 : index
          %37 = arith.addi %34, %c1 : index
          %38 = arith.addi %37, %35 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          %39 = loom.alloc [64] on @L1 : memref<64xf16>
          %40 = loom.semaphore_take %39 : memref<64xf16> -> memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25 : memref<64xf16>) outs(%40 : memref<64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %89 = arith.mulf %in, %cst_0 : f16
            %90 = math.powf %cst, %89 : f16
            linalg.yield %90 : f16
          }
          %41 = arith.divui %22, %c64 : index
          %42 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %43 = loom.semaphore_take %42 : memref<64x64xf16> -> memref<64x64xf16>
          %44 = arith.muli %arg12, %c65536 : index
          %45 = arith.addi %26, %44 : index
          %46 = arith.muli %41, %c64 overflow<nsw> : index
          %47 = arith.addi %45, %46 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          %48 = arith.addi %32, %c1 : index
          loom.copy %reinterpret_cast_1, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%48, %38]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          %49 = arith.muli %arg10, %c32 : index
          %50 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %51 = loom.semaphore_take %50 : memref<32x64xf16> -> memref<32x64xf16>
          %52 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %53 = arith.muli %arg12, %c1048576 : index
          %54 = arith.addi %52, %53 : index
          %55 = arith.muli %arg8, %c131072 : index
          %56 = arith.addi %54, %55 : index
          %57 = arith.muli %arg10, %c2048 : index
          %58 = arith.addi %56, %57 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%58], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          %59 = arith.addi %arg10, %34 : index
          %60 = arith.addi %59, %35 : index
          loom.copy %reinterpret_cast_2, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %60], LR : [%48, %60]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.transpose ins(%51 : memref<32x64xf16>) outs(%62 : memref<64x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %51 : memref<32x64xf16>
          %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          %65 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          loom.matmul ins(%43, %62 : memref<64x64xf16>, memref<64x32xf16>) outs(%64 : memref<64x32xf16>)
          loom.semaphore_give %62 : memref<64x32xf16>
          loom.semaphore_give %43 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %40 : memref<64x32xf16>, memref<64xf16>) outs(%65 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %89 = arith.mulf %in, %in_6 : f16
            linalg.yield %89 : f16
          }
          loom.semaphore_give %64 : memref<64x32xf16>
          loom.semaphore_give %40 : memref<64xf16>
          %66 = arith.addi %21, %c1 : index
          %67 = arith.muli %66, %c64 : index
          %68 = arith.ceildivui %67, %c64 : index
          %69 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %70 = loom.semaphore_take %69 : memref<64x64xf16> -> memref<64x64xf16>
          %71 = loom.alloc [64] on @L1 : memref<64xf16>
          %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
          %73 = loom.alloc [64] on @L1 : memref<64xf16>
          %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %75 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %76 = loom.semaphore_take %75 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %68 step %c1 {
            %89 = arith.muli %arg14, %c64 : index
            %90 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %91 = arith.muli %arg12, %c262144 : index
            %92 = arith.addi %90, %91 : index
            %93 = arith.muli %41, %c65536 overflow<nsw> : index
            %94 = arith.addi %92, %93 : index
            %95 = arith.muli %21, %c16384 : index
            %96 = arith.addi %94, %95 : index
            %97 = arith.addi %96, %89 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%97], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %98 = arith.addi %30, %89 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%98], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%48, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%98], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%48, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %25, %72, %74 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%70 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_10: f16, %in_11: f16, %in_12: f16, %out: f16):
              %105 = arith.mulf %in_11, %cst_0 : f16
              %106 = arith.mulf %in_10, %cst_0 : f16
              %107 = arith.subf %106, %105 : f16
              %108 = math.powf %cst, %107 : f16
              %109 = arith.mulf %in, %108 : f16
              %110 = arith.mulf %109, %in_12 : f16
              linalg.yield %110 : f16
            }
            loom.semaphore_give %74 : memref<64xf16>
            loom.semaphore_give %72 : memref<64xf16>
            %99 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %100 = arith.muli %arg12, %c4194304 : index
            %101 = arith.addi %99, %100 : index
            %102 = arith.muli %arg8, %c2048 : index
            %103 = arith.addi %101, %102 : index
            %104 = arith.addi %103, %49 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%104], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %60], LR : [%48, %60]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%70, %76 : memref<64x64xf16>, memref<64x32xf16>) outs(%65 : memref<64x32xf16>)
            loom.semaphore_give %76 : memref<64x32xf16>
            loom.semaphore_give %70 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %77 = loom.alloc [1] on @L1 : memref<f16>
          %78 = loom.semaphore_take %77 : memref<f16> -> memref<f16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %79 = arith.addi %35, %c3 : index
          loom.copy %reinterpret_cast_3, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %35], LR : [%c7, %79]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          %82 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %83 = arith.muli %arg12, %c4194304 : index
          %84 = arith.addi %82, %83 : index
          %85 = arith.muli %arg8, %c2048 : index
          %86 = arith.addi %84, %85 : index
          %87 = arith.addi %86, %49 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %60], LR : [%48, %60]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %81, %78 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%81 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
            %89 = arith.mulf %in_6, %in_7 : f16
            %90 = arith.addf %in, %89 : f16
            linalg.yield %90 : f16
          }
          loom.semaphore_give %78 : memref<f16>
          loom.semaphore_give %65 : memref<64x32xf16>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%87], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %88 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%81 : memref<64x32xf16>) outs(%88 : memref<64x32xf16>)
          loom.copy %88, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%32, %60], LR : [%48, %60]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %88 : memref<64x32xf16>
          loom.semaphore_give %81 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c16384 = arith.constant 16384 : index
      %c262144 = arith.constant 262144 : index
      %c2048 = arith.constant 2048 : index
      %c1048576 = arith.constant 1048576 : index
      %c1024 = arith.constant 1024 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 1.442380e+00 : f16
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c4, %c2, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        %20 = arith.muli %arg8, %c32 : index
        %21 = arith.muli %arg9, %c64 : index
        %22 = loom.alloc [64] on @L1 : memref<64xf16>
        %23 = loom.semaphore_take %22 : memref<64xf16> -> memref<64xf16>
        %24 = arith.muli %arg11, %c131072 overflow<nsw> : index
        %25 = arith.muli %arg8, %c65536 : index
        %26 = arith.addi %24, %25 : index
        %27 = arith.muli %arg12, %c1024 : index
        %28 = arith.addi %26, %27 : index
        %29 = arith.addi %28, %21 : index
        %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
        %30 = arith.muli %arg11, %c4 : index
        %31 = arith.addi %arg9, %30 : index
        %32 = arith.muli %arg12, %c2 : index
        %33 = arith.muli %arg8, %c4 : index
        %34 = arith.addi %32, %33 : index
        %35 = arith.addi %32, %c1 : index
        %36 = arith.addi %35, %33 : index
        loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
        %37 = loom.alloc [64] on @L1 : memref<64xf16>
        %38 = loom.semaphore_take %37 : memref<64xf16> -> memref<64xf16>
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%23 : memref<64xf16>) outs(%38 : memref<64xf16>) {
        ^bb0(%in: f16, %out: f16):
          %87 = arith.mulf %in, %cst_0 : f16
          %88 = math.powf %cst, %87 : f16
          linalg.yield %88 : f16
        }
        %39 = arith.divui %20, %c64 : index
        %40 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %41 = loom.semaphore_take %40 : memref<64x64xf16> -> memref<64x64xf16>
        %42 = arith.muli %arg12, %c65536 : index
        %43 = arith.addi %24, %42 : index
        %44 = arith.muli %39, %c64 overflow<nsw> : index
        %45 = arith.addi %43, %44 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%45], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
        %46 = arith.addi %30, %c3 : index
        loom.copy %reinterpret_cast_1, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%46, %36]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
        %47 = arith.muli %arg10, %c32 : index
        %48 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
        %49 = loom.semaphore_take %48 : memref<32x64xf16> -> memref<32x64xf16>
        %50 = arith.muli %arg11, %c2097152 overflow<nsw> : index
        %51 = arith.muli %arg12, %c1048576 : index
        %52 = arith.addi %50, %51 : index
        %53 = arith.muli %arg8, %c131072 : index
        %54 = arith.addi %52, %53 : index
        %55 = arith.muli %arg10, %c2048 : index
        %56 = arith.addi %54, %55 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%56], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %57 = arith.addi %arg10, %32 : index
        %58 = arith.addi %57, %33 : index
        loom.copy %reinterpret_cast_2, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %58], LR : [%46, %58]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %59 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %60 = loom.semaphore_take %59 : memref<64x32xf16> -> memref<64x32xf16>
        linalg.transpose ins(%49 : memref<32x64xf16>) outs(%60 : memref<64x32xf16>) permutation = [1, 0] 
        loom.semaphore_give %49 : memref<32x64xf16>
        %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
        %63 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
        loom.matmul ins(%41, %60 : memref<64x64xf16>, memref<64x32xf16>) outs(%62 : memref<64x32xf16>)
        loom.semaphore_give %60 : memref<64x32xf16>
        loom.semaphore_give %41 : memref<64x64xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%62, %38 : memref<64x32xf16>, memref<64xf16>) outs(%63 : memref<64x32xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %87 = arith.mulf %in, %in_6 : f16
          linalg.yield %87 : f16
        }
        loom.semaphore_give %62 : memref<64x32xf16>
        loom.semaphore_give %38 : memref<64xf16>
        %64 = arith.addi %arg9, %c1 : index
        %65 = arith.muli %64, %c64 : index
        %66 = arith.ceildivui %65, %c64 : index
        %67 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %68 = loom.semaphore_take %67 : memref<64x64xf16> -> memref<64x64xf16>
        %69 = loom.alloc [64] on @L1 : memref<64xf16>
        %70 = loom.semaphore_take %69 : memref<64xf16> -> memref<64xf16>
        %71 = loom.alloc [64] on @L1 : memref<64xf16>
        %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
        %73 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %74 = loom.semaphore_take %73 : memref<64x32xf16> -> memref<64x32xf16>
        scf.for %arg13 = %c0 to %66 step %c1 {
          %87 = arith.muli %arg13, %c64 : index
          %88 = arith.muli %arg11, %c524288 overflow<nsw> : index
          %89 = arith.muli %arg12, %c262144 : index
          %90 = arith.addi %88, %89 : index
          %91 = arith.muli %39, %c65536 overflow<nsw> : index
          %92 = arith.addi %90, %91 : index
          %93 = arith.muli %arg9, %c16384 : index
          %94 = arith.addi %92, %93 : index
          %95 = arith.addi %94, %87 : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%95], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_6, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
          %96 = arith.addi %28, %87 : index
          %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%96], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_7, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%46, %36]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%96], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_8, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%46, %36]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%68, %23, %70, %72 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%68 : memref<64x64xf16>) {
          ^bb0(%in: f16, %in_10: f16, %in_11: f16, %in_12: f16, %out: f16):
            %103 = arith.mulf %in_11, %cst_0 : f16
            %104 = arith.mulf %in_10, %cst_0 : f16
            %105 = arith.subf %104, %103 : f16
            %106 = math.powf %cst, %105 : f16
            %107 = arith.mulf %in, %106 : f16
            %108 = arith.mulf %107, %in_12 : f16
            linalg.yield %108 : f16
          }
          loom.semaphore_give %72 : memref<64xf16>
          loom.semaphore_give %70 : memref<64xf16>
          %97 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %98 = arith.muli %arg12, %c4194304 : index
          %99 = arith.addi %97, %98 : index
          %100 = arith.muli %arg8, %c2048 : index
          %101 = arith.addi %99, %100 : index
          %102 = arith.addi %101, %47 : index
          %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%102], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_9, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %58], LR : [%46, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.matmul ins(%68, %74 : memref<64x64xf16>, memref<64x32xf16>) outs(%63 : memref<64x32xf16>)
          loom.semaphore_give %74 : memref<64x32xf16>
          loom.semaphore_give %68 : memref<64x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        loom.semaphore_give %23 : memref<64xf16>
        %75 = loom.alloc [1] on @L1 : memref<f16>
        %76 = loom.semaphore_take %75 : memref<f16> -> memref<f16>
        %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%20], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
        %77 = arith.addi %33, %c3 : index
        loom.copy %reinterpret_cast_3, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %33], LR : [%c7, %77]) : memref<f16, strided<[], offset: ?>> to memref<f16>
        %78 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %79 = loom.semaphore_take %78 : memref<64x32xf16> -> memref<64x32xf16>
        %80 = arith.muli %arg11, %c8388608 overflow<nsw> : index
        %81 = arith.muli %arg12, %c4194304 : index
        %82 = arith.addi %80, %81 : index
        %83 = arith.muli %arg8, %c2048 : index
        %84 = arith.addi %82, %83 : index
        %85 = arith.addi %84, %47 : index
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%85], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_4, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %58], LR : [%46, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %79, %76 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%79 : memref<64x32xf16>) {
        ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
          %87 = arith.mulf %in_6, %in_7 : f16
          %88 = arith.addf %in, %87 : f16
          linalg.yield %88 : f16
        }
        loom.semaphore_give %76 : memref<f16>
        loom.semaphore_give %63 : memref<64x32xf16>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%85], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
        %86 = loom.semaphore_take %78 : memref<64x32xf16> -> memref<64x32xf16>
        loom.sync ins(%79 : memref<64x32xf16>) outs(%86 : memref<64x32xf16>)
        loom.copy %86, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%30, %58], LR : [%46, %58]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %86 : memref<64x32xf16>
        loom.semaphore_give %79 : memref<64x32xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c16384 = arith.constant 16384 : index
      %c262144 = arith.constant 262144 : index
      %c2048 = arith.constant 2048 : index
      %c1048576 = arith.constant 1048576 : index
      %c1024 = arith.constant 1024 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 1.442380e+00 : f16
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c4, %c2, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        %20 = arith.muli %arg8, %c32 : index
        %21 = arith.muli %arg9, %c64 : index
        %22 = loom.alloc [64] on @L1 : memref<64xf16>
        %23 = loom.semaphore_take %22 : memref<64xf16> -> memref<64xf16>
        %24 = arith.muli %arg11, %c131072 overflow<nsw> : index
        %25 = arith.muli %arg8, %c65536 : index
        %26 = arith.addi %24, %25 : index
        %27 = arith.muli %arg12, %c1024 : index
        %28 = arith.addi %26, %27 : index
        %29 = arith.addi %28, %21 : index
        %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
        %30 = arith.muli %arg12, %c4 : index
        %31 = arith.addi %arg9, %30 : index
        %32 = arith.muli %arg11, %c2 : index
        %33 = arith.muli %arg8, %c4 : index
        %34 = arith.addi %32, %33 : index
        %35 = arith.addi %32, %c1 : index
        %36 = arith.addi %35, %33 : index
        loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
        %37 = loom.alloc [64] on @L1 : memref<64xf16>
        %38 = loom.semaphore_take %37 : memref<64xf16> -> memref<64xf16>
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%23 : memref<64xf16>) outs(%38 : memref<64xf16>) {
        ^bb0(%in: f16, %out: f16):
          %87 = arith.mulf %in, %cst_0 : f16
          %88 = math.powf %cst, %87 : f16
          linalg.yield %88 : f16
        }
        %39 = arith.divui %20, %c64 : index
        %40 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %41 = loom.semaphore_take %40 : memref<64x64xf16> -> memref<64x64xf16>
        %42 = arith.muli %arg12, %c65536 : index
        %43 = arith.addi %24, %42 : index
        %44 = arith.muli %39, %c64 overflow<nsw> : index
        %45 = arith.addi %43, %44 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%45], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
        %46 = arith.addi %30, %c3 : index
        loom.copy %reinterpret_cast_1, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%46, %36]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
        %47 = arith.muli %arg10, %c32 : index
        %48 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
        %49 = loom.semaphore_take %48 : memref<32x64xf16> -> memref<32x64xf16>
        %50 = arith.muli %arg11, %c2097152 overflow<nsw> : index
        %51 = arith.muli %arg12, %c1048576 : index
        %52 = arith.addi %50, %51 : index
        %53 = arith.muli %arg8, %c131072 : index
        %54 = arith.addi %52, %53 : index
        %55 = arith.muli %arg10, %c2048 : index
        %56 = arith.addi %54, %55 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%56], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %57 = arith.addi %arg10, %32 : index
        %58 = arith.addi %57, %33 : index
        loom.copy %reinterpret_cast_2, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %58], LR : [%46, %58]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %59 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %60 = loom.semaphore_take %59 : memref<64x32xf16> -> memref<64x32xf16>
        linalg.transpose ins(%49 : memref<32x64xf16>) outs(%60 : memref<64x32xf16>) permutation = [1, 0] 
        loom.semaphore_give %49 : memref<32x64xf16>
        %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
        %63 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
        loom.matmul ins(%41, %60 : memref<64x64xf16>, memref<64x32xf16>) outs(%62 : memref<64x32xf16>)
        loom.semaphore_give %60 : memref<64x32xf16>
        loom.semaphore_give %41 : memref<64x64xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%62, %38 : memref<64x32xf16>, memref<64xf16>) outs(%63 : memref<64x32xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %87 = arith.mulf %in, %in_6 : f16
          linalg.yield %87 : f16
        }
        loom.semaphore_give %62 : memref<64x32xf16>
        loom.semaphore_give %38 : memref<64xf16>
        %64 = arith.addi %arg9, %c1 : index
        %65 = arith.muli %64, %c64 : index
        %66 = arith.ceildivui %65, %c64 : index
        %67 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %68 = loom.semaphore_take %67 : memref<64x64xf16> -> memref<64x64xf16>
        %69 = loom.alloc [64] on @L1 : memref<64xf16>
        %70 = loom.semaphore_take %69 : memref<64xf16> -> memref<64xf16>
        %71 = loom.alloc [64] on @L1 : memref<64xf16>
        %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
        %73 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %74 = loom.semaphore_take %73 : memref<64x32xf16> -> memref<64x32xf16>
        scf.for %arg13 = %c0 to %66 step %c1 {
          %87 = arith.muli %arg13, %c64 : index
          %88 = arith.muli %arg11, %c524288 overflow<nsw> : index
          %89 = arith.muli %arg12, %c262144 : index
          %90 = arith.addi %88, %89 : index
          %91 = arith.muli %39, %c65536 overflow<nsw> : index
          %92 = arith.addi %90, %91 : index
          %93 = arith.muli %arg9, %c16384 : index
          %94 = arith.addi %92, %93 : index
          %95 = arith.addi %94, %87 : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%95], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_6, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
          %96 = arith.addi %28, %87 : index
          %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%96], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_7, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%46, %36]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%96], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_8, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%46, %36]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%68, %23, %70, %72 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%68 : memref<64x64xf16>) {
          ^bb0(%in: f16, %in_10: f16, %in_11: f16, %in_12: f16, %out: f16):
            %103 = arith.mulf %in_11, %cst_0 : f16
            %104 = arith.mulf %in_10, %cst_0 : f16
            %105 = arith.subf %104, %103 : f16
            %106 = math.powf %cst, %105 : f16
            %107 = arith.mulf %in, %106 : f16
            %108 = arith.mulf %107, %in_12 : f16
            linalg.yield %108 : f16
          }
          loom.semaphore_give %72 : memref<64xf16>
          loom.semaphore_give %70 : memref<64xf16>
          %97 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %98 = arith.muli %arg12, %c4194304 : index
          %99 = arith.addi %97, %98 : index
          %100 = arith.muli %arg8, %c2048 : index
          %101 = arith.addi %99, %100 : index
          %102 = arith.addi %101, %47 : index
          %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%102], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_9, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %58], LR : [%46, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.matmul ins(%68, %74 : memref<64x64xf16>, memref<64x32xf16>) outs(%63 : memref<64x32xf16>)
          loom.semaphore_give %74 : memref<64x32xf16>
          loom.semaphore_give %68 : memref<64x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        loom.semaphore_give %23 : memref<64xf16>
        %75 = loom.alloc [1] on @L1 : memref<f16>
        %76 = loom.semaphore_take %75 : memref<f16> -> memref<f16>
        %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%20], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
        %77 = arith.addi %33, %c3 : index
        loom.copy %reinterpret_cast_3, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %33], LR : [%c7, %77]) : memref<f16, strided<[], offset: ?>> to memref<f16>
        %78 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %79 = loom.semaphore_take %78 : memref<64x32xf16> -> memref<64x32xf16>
        %80 = arith.muli %arg11, %c8388608 overflow<nsw> : index
        %81 = arith.muli %arg12, %c4194304 : index
        %82 = arith.addi %80, %81 : index
        %83 = arith.muli %arg8, %c2048 : index
        %84 = arith.addi %82, %83 : index
        %85 = arith.addi %84, %47 : index
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%85], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_4, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %58], LR : [%46, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %79, %76 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%79 : memref<64x32xf16>) {
        ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
          %87 = arith.mulf %in_6, %in_7 : f16
          %88 = arith.addf %in, %87 : f16
          linalg.yield %88 : f16
        }
        loom.semaphore_give %76 : memref<f16>
        loom.semaphore_give %63 : memref<64x32xf16>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%85], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
        %86 = loom.semaphore_take %78 : memref<64x32xf16> -> memref<64x32xf16>
        loom.sync ins(%79 : memref<64x32xf16>) outs(%86 : memref<64x32xf16>)
        loom.copy %86, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%30, %58], LR : [%46, %58]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %86 : memref<64x32xf16>
        loom.semaphore_give %79 : memref<64x32xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c16384 = arith.constant 16384 : index
      %c262144 = arith.constant 262144 : index
      %c2048 = arith.constant 2048 : index
      %c1048576 = arith.constant 1048576 : index
      %c1024 = arith.constant 1024 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 1.442380e+00 : f16
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c2, %c4) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg9, %20 : index
          %22 = arith.muli %arg8, %c32 : index
          %23 = arith.muli %21, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %27 = arith.muli %arg8, %c65536 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg11, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg8, %c4 : index
          %35 = arith.addi %33, %34 : index
          %36 = arith.muli %arg12, %c2 : index
          %37 = arith.addi %36, %c1 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %36], LR : [%35, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          %38 = loom.alloc [64] on @L1 : memref<64xf16>
          %39 = loom.semaphore_take %38 : memref<64xf16> -> memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25 : memref<64xf16>) outs(%39 : memref<64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %89 = arith.mulf %in, %cst_0 : f16
            %90 = math.powf %cst, %89 : f16
            linalg.yield %90 : f16
          }
          %40 = arith.divui %22, %c64 : index
          %41 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %42 = loom.semaphore_take %41 : memref<64x64xf16> -> memref<64x64xf16>
          %43 = arith.muli %arg12, %c65536 : index
          %44 = arith.addi %26, %43 : index
          %45 = arith.muli %40, %c64 overflow<nsw> : index
          %46 = arith.addi %44, %45 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          %47 = arith.addi %32, %34 : index
          %48 = arith.addi %32, %c1 : index
          %49 = arith.addi %48, %34 : index
          loom.copy %reinterpret_cast_1, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%47, %36], LR : [%49, %37]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          %50 = arith.muli %arg10, %c32 : index
          %51 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %52 = loom.semaphore_take %51 : memref<32x64xf16> -> memref<32x64xf16>
          %53 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %54 = arith.muli %arg12, %c1048576 : index
          %55 = arith.addi %53, %54 : index
          %56 = arith.muli %arg8, %c131072 : index
          %57 = arith.addi %55, %56 : index
          %58 = arith.muli %arg10, %c2048 : index
          %59 = arith.addi %57, %58 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          %60 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_2, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.transpose ins(%52 : memref<32x64xf16>) outs(%62 : memref<64x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %52 : memref<32x64xf16>
          %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          %65 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          loom.matmul ins(%42, %62 : memref<64x64xf16>, memref<64x32xf16>) outs(%64 : memref<64x32xf16>)
          loom.semaphore_give %62 : memref<64x32xf16>
          loom.semaphore_give %42 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %39 : memref<64x32xf16>, memref<64xf16>) outs(%65 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %89 = arith.mulf %in, %in_6 : f16
            linalg.yield %89 : f16
          }
          loom.semaphore_give %64 : memref<64x32xf16>
          loom.semaphore_give %39 : memref<64xf16>
          %66 = arith.addi %21, %c1 : index
          %67 = arith.muli %66, %c64 : index
          %68 = arith.ceildivui %67, %c64 : index
          %69 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %70 = loom.semaphore_take %69 : memref<64x64xf16> -> memref<64x64xf16>
          %71 = loom.alloc [64] on @L1 : memref<64xf16>
          %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
          %73 = loom.alloc [64] on @L1 : memref<64xf16>
          %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %75 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %76 = loom.semaphore_take %75 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %68 step %c1 {
            %89 = arith.muli %arg14, %c64 : index
            %90 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %91 = arith.muli %arg12, %c262144 : index
            %92 = arith.addi %90, %91 : index
            %93 = arith.muli %40, %c65536 overflow<nsw> : index
            %94 = arith.addi %92, %93 : index
            %95 = arith.muli %21, %c16384 : index
            %96 = arith.addi %94, %95 : index
            %97 = arith.addi %96, %89 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%97], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %36], LR : [%35, %37]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %98 = arith.addi %30, %89 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%98], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%47, %36], LR : [%49, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%98], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%47, %36], LR : [%49, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %25, %72, %74 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%70 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_10: f16, %in_11: f16, %in_12: f16, %out: f16):
              %105 = arith.mulf %in_11, %cst_0 : f16
              %106 = arith.mulf %in_10, %cst_0 : f16
              %107 = arith.subf %106, %105 : f16
              %108 = math.powf %cst, %107 : f16
              %109 = arith.mulf %in, %108 : f16
              %110 = arith.mulf %109, %in_12 : f16
              linalg.yield %110 : f16
            }
            loom.semaphore_give %74 : memref<64xf16>
            loom.semaphore_give %72 : memref<64xf16>
            %99 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %100 = arith.muli %arg12, %c4194304 : index
            %101 = arith.addi %99, %100 : index
            %102 = arith.muli %arg8, %c2048 : index
            %103 = arith.addi %101, %102 : index
            %104 = arith.addi %103, %50 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%104], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%70, %76 : memref<64x64xf16>, memref<64x32xf16>) outs(%65 : memref<64x32xf16>)
            loom.semaphore_give %76 : memref<64x32xf16>
            loom.semaphore_give %70 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %77 = loom.alloc [1] on @L1 : memref<f16>
          %78 = loom.semaphore_take %77 : memref<f16> -> memref<f16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %79 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_3, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%79, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          %82 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %83 = arith.muli %arg12, %c4194304 : index
          %84 = arith.addi %82, %83 : index
          %85 = arith.muli %arg8, %c2048 : index
          %86 = arith.addi %84, %85 : index
          %87 = arith.addi %86, %50 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %81, %78 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%81 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
            %89 = arith.mulf %in_6, %in_7 : f16
            %90 = arith.addf %in, %89 : f16
            linalg.yield %90 : f16
          }
          loom.semaphore_give %78 : memref<f16>
          loom.semaphore_give %65 : memref<64x32xf16>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%87], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %88 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%81 : memref<64x32xf16>) outs(%88 : memref<64x32xf16>)
          loom.copy %88, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %88 : memref<64x32xf16>
          loom.semaphore_give %81 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c16384 = arith.constant 16384 : index
      %c262144 = arith.constant 262144 : index
      %c2048 = arith.constant 2048 : index
      %c1048576 = arith.constant 1048576 : index
      %c1024 = arith.constant 1024 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 1.442380e+00 : f16
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c4, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg9, %20 : index
          %22 = arith.muli %arg8, %c32 : index
          %23 = arith.muli %21, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %27 = arith.muli %arg8, %c65536 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg12, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg8, %c4 : index
          %35 = arith.addi %33, %34 : index
          %36 = arith.muli %arg11, %c2 : index
          %37 = arith.addi %36, %c1 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %36], LR : [%35, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          %38 = loom.alloc [64] on @L1 : memref<64xf16>
          %39 = loom.semaphore_take %38 : memref<64xf16> -> memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25 : memref<64xf16>) outs(%39 : memref<64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %89 = arith.mulf %in, %cst_0 : f16
            %90 = math.powf %cst, %89 : f16
            linalg.yield %90 : f16
          }
          %40 = arith.divui %22, %c64 : index
          %41 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %42 = loom.semaphore_take %41 : memref<64x64xf16> -> memref<64x64xf16>
          %43 = arith.muli %arg12, %c65536 : index
          %44 = arith.addi %26, %43 : index
          %45 = arith.muli %40, %c64 overflow<nsw> : index
          %46 = arith.addi %44, %45 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          %47 = arith.addi %32, %34 : index
          %48 = arith.addi %32, %c1 : index
          %49 = arith.addi %48, %34 : index
          loom.copy %reinterpret_cast_1, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%47, %36], LR : [%49, %37]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          %50 = arith.muli %arg10, %c32 : index
          %51 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %52 = loom.semaphore_take %51 : memref<32x64xf16> -> memref<32x64xf16>
          %53 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %54 = arith.muli %arg12, %c1048576 : index
          %55 = arith.addi %53, %54 : index
          %56 = arith.muli %arg8, %c131072 : index
          %57 = arith.addi %55, %56 : index
          %58 = arith.muli %arg10, %c2048 : index
          %59 = arith.addi %57, %58 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          %60 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_2, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.transpose ins(%52 : memref<32x64xf16>) outs(%62 : memref<64x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %52 : memref<32x64xf16>
          %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          %65 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          loom.matmul ins(%42, %62 : memref<64x64xf16>, memref<64x32xf16>) outs(%64 : memref<64x32xf16>)
          loom.semaphore_give %62 : memref<64x32xf16>
          loom.semaphore_give %42 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %39 : memref<64x32xf16>, memref<64xf16>) outs(%65 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %89 = arith.mulf %in, %in_6 : f16
            linalg.yield %89 : f16
          }
          loom.semaphore_give %64 : memref<64x32xf16>
          loom.semaphore_give %39 : memref<64xf16>
          %66 = arith.addi %21, %c1 : index
          %67 = arith.muli %66, %c64 : index
          %68 = arith.ceildivui %67, %c64 : index
          %69 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %70 = loom.semaphore_take %69 : memref<64x64xf16> -> memref<64x64xf16>
          %71 = loom.alloc [64] on @L1 : memref<64xf16>
          %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
          %73 = loom.alloc [64] on @L1 : memref<64xf16>
          %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %75 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %76 = loom.semaphore_take %75 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %68 step %c1 {
            %89 = arith.muli %arg14, %c64 : index
            %90 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %91 = arith.muli %arg12, %c262144 : index
            %92 = arith.addi %90, %91 : index
            %93 = arith.muli %40, %c65536 overflow<nsw> : index
            %94 = arith.addi %92, %93 : index
            %95 = arith.muli %21, %c16384 : index
            %96 = arith.addi %94, %95 : index
            %97 = arith.addi %96, %89 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%97], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %36], LR : [%35, %37]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %98 = arith.addi %30, %89 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%98], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%47, %36], LR : [%49, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%98], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%47, %36], LR : [%49, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %25, %72, %74 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%70 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_10: f16, %in_11: f16, %in_12: f16, %out: f16):
              %105 = arith.mulf %in_11, %cst_0 : f16
              %106 = arith.mulf %in_10, %cst_0 : f16
              %107 = arith.subf %106, %105 : f16
              %108 = math.powf %cst, %107 : f16
              %109 = arith.mulf %in, %108 : f16
              %110 = arith.mulf %109, %in_12 : f16
              linalg.yield %110 : f16
            }
            loom.semaphore_give %74 : memref<64xf16>
            loom.semaphore_give %72 : memref<64xf16>
            %99 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %100 = arith.muli %arg12, %c4194304 : index
            %101 = arith.addi %99, %100 : index
            %102 = arith.muli %arg8, %c2048 : index
            %103 = arith.addi %101, %102 : index
            %104 = arith.addi %103, %50 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%104], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%70, %76 : memref<64x64xf16>, memref<64x32xf16>) outs(%65 : memref<64x32xf16>)
            loom.semaphore_give %76 : memref<64x32xf16>
            loom.semaphore_give %70 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %77 = loom.alloc [1] on @L1 : memref<f16>
          %78 = loom.semaphore_take %77 : memref<f16> -> memref<f16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %79 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_3, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%79, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          %82 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %83 = arith.muli %arg12, %c4194304 : index
          %84 = arith.addi %82, %83 : index
          %85 = arith.muli %arg8, %c2048 : index
          %86 = arith.addi %84, %85 : index
          %87 = arith.addi %86, %50 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %81, %78 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%81 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
            %89 = arith.mulf %in_6, %in_7 : f16
            %90 = arith.addf %in, %89 : f16
            linalg.yield %90 : f16
          }
          loom.semaphore_give %78 : memref<f16>
          loom.semaphore_give %65 : memref<64x32xf16>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%87], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %88 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%81 : memref<64x32xf16>) outs(%88 : memref<64x32xf16>)
          loom.copy %88, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %88 : memref<64x32xf16>
          loom.semaphore_give %81 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c16384 = arith.constant 16384 : index
      %c262144 = arith.constant 262144 : index
      %c2048 = arith.constant 2048 : index
      %c1048576 = arith.constant 1048576 : index
      %c1024 = arith.constant 1024 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 1.442380e+00 : f16
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c4, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg9, %20 : index
          %22 = arith.muli %arg8, %c32 : index
          %23 = arith.muli %21, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %27 = arith.muli %arg8, %c65536 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg11, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg8, %c4 : index
          %35 = arith.addi %33, %34 : index
          %36 = arith.muli %arg12, %c4 : index
          %37 = arith.addi %36, %c3 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%35, %36], LR : [%35, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          %38 = loom.alloc [64] on @L1 : memref<64xf16>
          %39 = loom.semaphore_take %38 : memref<64xf16> -> memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25 : memref<64xf16>) outs(%39 : memref<64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %89 = arith.mulf %in, %cst_0 : f16
            %90 = math.powf %cst, %89 : f16
            linalg.yield %90 : f16
          }
          %40 = arith.divui %22, %c64 : index
          %41 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %42 = loom.semaphore_take %41 : memref<64x64xf16> -> memref<64x64xf16>
          %43 = arith.muli %arg12, %c65536 : index
          %44 = arith.addi %26, %43 : index
          %45 = arith.muli %40, %c64 overflow<nsw> : index
          %46 = arith.addi %44, %45 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          %47 = arith.addi %32, %34 : index
          %48 = arith.addi %32, %c1 : index
          %49 = arith.addi %48, %34 : index
          loom.copy %reinterpret_cast_1, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%47, %36], LR : [%49, %37]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          %50 = arith.muli %arg10, %c32 : index
          %51 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %52 = loom.semaphore_take %51 : memref<32x64xf16> -> memref<32x64xf16>
          %53 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %54 = arith.muli %arg12, %c1048576 : index
          %55 = arith.addi %53, %54 : index
          %56 = arith.muli %arg8, %c131072 : index
          %57 = arith.addi %55, %56 : index
          %58 = arith.muli %arg10, %c2048 : index
          %59 = arith.addi %57, %58 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          %60 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_2, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.transpose ins(%52 : memref<32x64xf16>) outs(%62 : memref<64x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %52 : memref<32x64xf16>
          %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          %65 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          loom.matmul ins(%42, %62 : memref<64x64xf16>, memref<64x32xf16>) outs(%64 : memref<64x32xf16>)
          loom.semaphore_give %62 : memref<64x32xf16>
          loom.semaphore_give %42 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %39 : memref<64x32xf16>, memref<64xf16>) outs(%65 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %89 = arith.mulf %in, %in_6 : f16
            linalg.yield %89 : f16
          }
          loom.semaphore_give %64 : memref<64x32xf16>
          loom.semaphore_give %39 : memref<64xf16>
          %66 = arith.addi %21, %c1 : index
          %67 = arith.muli %66, %c64 : index
          %68 = arith.ceildivui %67, %c64 : index
          %69 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %70 = loom.semaphore_take %69 : memref<64x64xf16> -> memref<64x64xf16>
          %71 = loom.alloc [64] on @L1 : memref<64xf16>
          %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
          %73 = loom.alloc [64] on @L1 : memref<64xf16>
          %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %75 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %76 = loom.semaphore_take %75 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %68 step %c1 {
            %89 = arith.muli %arg14, %c64 : index
            %90 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %91 = arith.muli %arg12, %c262144 : index
            %92 = arith.addi %90, %91 : index
            %93 = arith.muli %40, %c65536 overflow<nsw> : index
            %94 = arith.addi %92, %93 : index
            %95 = arith.muli %21, %c16384 : index
            %96 = arith.addi %94, %95 : index
            %97 = arith.addi %96, %89 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%97], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%35, %36], LR : [%35, %37]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %98 = arith.addi %30, %89 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%98], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%47, %36], LR : [%49, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%98], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%47, %36], LR : [%49, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %25, %72, %74 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%70 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_10: f16, %in_11: f16, %in_12: f16, %out: f16):
              %105 = arith.mulf %in_11, %cst_0 : f16
              %106 = arith.mulf %in_10, %cst_0 : f16
              %107 = arith.subf %106, %105 : f16
              %108 = math.powf %cst, %107 : f16
              %109 = arith.mulf %in, %108 : f16
              %110 = arith.mulf %109, %in_12 : f16
              linalg.yield %110 : f16
            }
            loom.semaphore_give %74 : memref<64xf16>
            loom.semaphore_give %72 : memref<64xf16>
            %99 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %100 = arith.muli %arg12, %c4194304 : index
            %101 = arith.addi %99, %100 : index
            %102 = arith.muli %arg8, %c2048 : index
            %103 = arith.addi %101, %102 : index
            %104 = arith.addi %103, %50 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%104], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%70, %76 : memref<64x64xf16>, memref<64x32xf16>) outs(%65 : memref<64x32xf16>)
            loom.semaphore_give %76 : memref<64x32xf16>
            loom.semaphore_give %70 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %77 = loom.alloc [1] on @L1 : memref<f16>
          %78 = loom.semaphore_take %77 : memref<f16> -> memref<f16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %79 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_3, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%79, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          %82 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %83 = arith.muli %arg12, %c4194304 : index
          %84 = arith.addi %82, %83 : index
          %85 = arith.muli %arg8, %c2048 : index
          %86 = arith.addi %84, %85 : index
          %87 = arith.addi %86, %50 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %81, %78 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%81 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
            %89 = arith.mulf %in_6, %in_7 : f16
            %90 = arith.addf %in, %89 : f16
            linalg.yield %90 : f16
          }
          loom.semaphore_give %78 : memref<f16>
          loom.semaphore_give %65 : memref<64x32xf16>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%87], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %88 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%81 : memref<64x32xf16>) outs(%88 : memref<64x32xf16>)
          loom.copy %88, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %88 : memref<64x32xf16>
          loom.semaphore_give %81 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c16384 = arith.constant 16384 : index
      %c262144 = arith.constant 262144 : index
      %c2048 = arith.constant 2048 : index
      %c1048576 = arith.constant 1048576 : index
      %c1024 = arith.constant 1024 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 1.442380e+00 : f16
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c4, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg9, %20 : index
          %22 = arith.muli %arg8, %c32 : index
          %23 = arith.muli %21, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %27 = arith.muli %arg8, %c65536 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg12, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg8, %c4 : index
          %35 = arith.addi %33, %34 : index
          %36 = arith.muli %arg11, %c4 : index
          %37 = arith.addi %36, %c3 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%35, %36], LR : [%35, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          %38 = loom.alloc [64] on @L1 : memref<64xf16>
          %39 = loom.semaphore_take %38 : memref<64xf16> -> memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25 : memref<64xf16>) outs(%39 : memref<64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %89 = arith.mulf %in, %cst_0 : f16
            %90 = math.powf %cst, %89 : f16
            linalg.yield %90 : f16
          }
          %40 = arith.divui %22, %c64 : index
          %41 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %42 = loom.semaphore_take %41 : memref<64x64xf16> -> memref<64x64xf16>
          %43 = arith.muli %arg12, %c65536 : index
          %44 = arith.addi %26, %43 : index
          %45 = arith.muli %40, %c64 overflow<nsw> : index
          %46 = arith.addi %44, %45 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          %47 = arith.addi %32, %34 : index
          %48 = arith.addi %32, %c1 : index
          %49 = arith.addi %48, %34 : index
          loom.copy %reinterpret_cast_1, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%47, %36], LR : [%49, %37]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          %50 = arith.muli %arg10, %c32 : index
          %51 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %52 = loom.semaphore_take %51 : memref<32x64xf16> -> memref<32x64xf16>
          %53 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %54 = arith.muli %arg12, %c1048576 : index
          %55 = arith.addi %53, %54 : index
          %56 = arith.muli %arg8, %c131072 : index
          %57 = arith.addi %55, %56 : index
          %58 = arith.muli %arg10, %c2048 : index
          %59 = arith.addi %57, %58 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          %60 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_2, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.transpose ins(%52 : memref<32x64xf16>) outs(%62 : memref<64x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %52 : memref<32x64xf16>
          %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          %65 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          loom.matmul ins(%42, %62 : memref<64x64xf16>, memref<64x32xf16>) outs(%64 : memref<64x32xf16>)
          loom.semaphore_give %62 : memref<64x32xf16>
          loom.semaphore_give %42 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %39 : memref<64x32xf16>, memref<64xf16>) outs(%65 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %89 = arith.mulf %in, %in_6 : f16
            linalg.yield %89 : f16
          }
          loom.semaphore_give %64 : memref<64x32xf16>
          loom.semaphore_give %39 : memref<64xf16>
          %66 = arith.addi %21, %c1 : index
          %67 = arith.muli %66, %c64 : index
          %68 = arith.ceildivui %67, %c64 : index
          %69 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %70 = loom.semaphore_take %69 : memref<64x64xf16> -> memref<64x64xf16>
          %71 = loom.alloc [64] on @L1 : memref<64xf16>
          %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
          %73 = loom.alloc [64] on @L1 : memref<64xf16>
          %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %75 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %76 = loom.semaphore_take %75 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %68 step %c1 {
            %89 = arith.muli %arg14, %c64 : index
            %90 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %91 = arith.muli %arg12, %c262144 : index
            %92 = arith.addi %90, %91 : index
            %93 = arith.muli %40, %c65536 overflow<nsw> : index
            %94 = arith.addi %92, %93 : index
            %95 = arith.muli %21, %c16384 : index
            %96 = arith.addi %94, %95 : index
            %97 = arith.addi %96, %89 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%97], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%35, %36], LR : [%35, %37]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %98 = arith.addi %30, %89 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%98], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%47, %36], LR : [%49, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%98], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%47, %36], LR : [%49, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %25, %72, %74 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%70 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_10: f16, %in_11: f16, %in_12: f16, %out: f16):
              %105 = arith.mulf %in_11, %cst_0 : f16
              %106 = arith.mulf %in_10, %cst_0 : f16
              %107 = arith.subf %106, %105 : f16
              %108 = math.powf %cst, %107 : f16
              %109 = arith.mulf %in, %108 : f16
              %110 = arith.mulf %109, %in_12 : f16
              linalg.yield %110 : f16
            }
            loom.semaphore_give %74 : memref<64xf16>
            loom.semaphore_give %72 : memref<64xf16>
            %99 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %100 = arith.muli %arg12, %c4194304 : index
            %101 = arith.addi %99, %100 : index
            %102 = arith.muli %arg8, %c2048 : index
            %103 = arith.addi %101, %102 : index
            %104 = arith.addi %103, %50 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%104], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%70, %76 : memref<64x64xf16>, memref<64x32xf16>) outs(%65 : memref<64x32xf16>)
            loom.semaphore_give %76 : memref<64x32xf16>
            loom.semaphore_give %70 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %77 = loom.alloc [1] on @L1 : memref<f16>
          %78 = loom.semaphore_take %77 : memref<f16> -> memref<f16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %79 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_3, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%79, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          %82 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %83 = arith.muli %arg12, %c4194304 : index
          %84 = arith.addi %82, %83 : index
          %85 = arith.muli %arg8, %c2048 : index
          %86 = arith.addi %84, %85 : index
          %87 = arith.addi %86, %50 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%87], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %81, %78 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%81 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
            %89 = arith.mulf %in_6, %in_7 : f16
            %90 = arith.addf %in, %89 : f16
            linalg.yield %90 : f16
          }
          loom.semaphore_give %78 : memref<f16>
          loom.semaphore_give %65 : memref<64x32xf16>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%87], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %88 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%81 : memref<64x32xf16>) outs(%88 : memref<64x32xf16>)
          loom.copy %88, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%47, %60], LR : [%49, %60]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %88 : memref<64x32xf16>
          loom.semaphore_give %81 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
}
