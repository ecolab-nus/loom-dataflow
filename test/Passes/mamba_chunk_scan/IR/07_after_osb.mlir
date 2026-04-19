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
      %cst = arith.constant 2.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
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
          %39 = arith.divui %22, %c64 : index
          %40 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %41 = loom.semaphore_take %40 : memref<64x64xf16> -> memref<64x64xf16>
          %42 = arith.muli %arg12, %c65536 : index
          %43 = arith.addi %26, %42 : index
          %44 = arith.muli %39, %c64 overflow<nsw> : index
          %45 = arith.addi %43, %44 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%45], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          %46 = arith.addi %32, %c1 : index
          loom.copy %reinterpret_cast_2, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%46, %38]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
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
          %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%56], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          %57 = arith.addi %arg10, %34 : index
          %58 = arith.addi %57, %35 : index
          loom.copy %reinterpret_cast_3, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %58], LR : [%46, %58]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %59 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %60 = loom.semaphore_take %59 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.transpose ins(%49 : memref<32x64xf16>) outs(%60 : memref<64x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %49 : memref<32x64xf16>
          %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          %63 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.fill ins(%cst_0 : f16) outs(%62 : memref<64x32xf16>)
          linalg.matmul ins(%41, %60 : memref<64x64xf16>, memref<64x32xf16>) outs(%62 : memref<64x32xf16>)
          loom.semaphore_give %60 : memref<64x32xf16>
          loom.semaphore_give %41 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%62, %25 : memref<64x32xf16>, memref<64xf16>) outs(%63 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_7: f16, %out: f16):
            %88 = arith.mulf %in_7, %cst_1 : f16
            %89 = math.powf %cst, %88 : f16
            %90 = arith.mulf %in, %89 : f16
            linalg.yield %90 : f16
          }
          loom.semaphore_give %62 : memref<64x32xf16>
          %64 = arith.addi %21, %c1 : index
          %65 = arith.muli %64, %c64 : index
          %66 = arith.ceildivui %65, %c64 : index
          %67 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %68 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          %69 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          %70 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %71 = loom.semaphore_take %70 : memref<64x64xf16> -> memref<64x64xf16>
          %72 = loom.alloc [64] on @L1 : memref<64xf16>
          %73 = loom.semaphore_take %72 : memref<64xf16> -> memref<64xf16>
          %74 = loom.alloc [64] on @L1 : memref<64xf16>
          %75 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
          %76 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %77 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %66 step %c1 {
            %88 = arith.muli %arg14, %c64 : index
            %89 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %90 = arith.muli %arg12, %c262144 : index
            %91 = arith.addi %89, %90 : index
            %92 = arith.muli %39, %c65536 overflow<nsw> : index
            %93 = arith.addi %91, %92 : index
            %94 = arith.muli %21, %c16384 : index
            %95 = arith.addi %93, %94 : index
            %96 = arith.addi %95, %88 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%96], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_7, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %97 = arith.addi %30, %88 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%97], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%46, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%97], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_9, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%46, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%71, %25, %73, %75 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%71 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
              %104 = arith.mulf %in_12, %cst_1 : f16
              %105 = arith.mulf %in_11, %cst_1 : f16
              %106 = arith.subf %105, %104 : f16
              %107 = math.powf %cst, %106 : f16
              %108 = arith.mulf %in, %107 : f16
              %109 = arith.mulf %108, %in_13 : f16
              linalg.yield %109 : f16
            }
            loom.semaphore_give %75 : memref<64xf16>
            loom.semaphore_give %73 : memref<64xf16>
            %98 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %99 = arith.muli %arg12, %c4194304 : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.muli %arg8, %c2048 : index
            %102 = arith.addi %100, %101 : index
            %103 = arith.addi %102, %47 : index
            %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_10, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %58], LR : [%46, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.fill ins(%cst_0 : f16) outs(%69 : memref<64x32xf16>)
            linalg.matmul ins(%71, %77 : memref<64x64xf16>, memref<64x32xf16>) outs(%69 : memref<64x32xf16>)
            loom.semaphore_give %77 : memref<64x32xf16>
            loom.semaphore_give %71 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %69 : memref<64x32xf16>, memref<64x32xf16>) outs(%63 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_11: f16, %out: f16):
              %104 = arith.addf %in, %in_11 : f16
              linalg.yield %104 : f16
            }
            loom.semaphore_give %69 : memref<64x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %78 = loom.alloc [1] on @L1 : memref<f16>
          %79 = loom.semaphore_take %78 : memref<f16> -> memref<f16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %80 = arith.addi %35, %c3 : index
          loom.copy %reinterpret_cast_4, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %35], LR : [%c7, %80]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %81 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %82 = arith.muli %arg12, %c4194304 : index
          %83 = arith.addi %81, %82 : index
          %84 = arith.muli %arg8, %c2048 : index
          %85 = arith.addi %83, %84 : index
          %86 = arith.addi %85, %47 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_5, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %58], LR : [%46, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %68, %79 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%68 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
            %88 = arith.mulf %in_7, %in_8 : f16
            %89 = arith.addf %in, %88 : f16
            linalg.yield %89 : f16
          }
          loom.semaphore_give %79 : memref<f16>
          loom.semaphore_give %63 : memref<64x32xf16>
          %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %87 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%68 : memref<64x32xf16>) outs(%87 : memref<64x32xf16>)
          loom.copy %87, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%32, %58], LR : [%46, %58]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %87 : memref<64x32xf16>
          loom.semaphore_give %68 : memref<64x32xf16>
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
      %cst = arith.constant 2.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
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
          %39 = arith.divui %22, %c64 : index
          %40 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %41 = loom.semaphore_take %40 : memref<64x64xf16> -> memref<64x64xf16>
          %42 = arith.muli %arg12, %c65536 : index
          %43 = arith.addi %26, %42 : index
          %44 = arith.muli %39, %c64 overflow<nsw> : index
          %45 = arith.addi %43, %44 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%45], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          %46 = arith.addi %32, %c1 : index
          loom.copy %reinterpret_cast_2, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%46, %38]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
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
          %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%56], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          %57 = arith.addi %arg10, %34 : index
          %58 = arith.addi %57, %35 : index
          loom.copy %reinterpret_cast_3, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %58], LR : [%46, %58]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %59 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %60 = loom.semaphore_take %59 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.transpose ins(%49 : memref<32x64xf16>) outs(%60 : memref<64x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %49 : memref<32x64xf16>
          %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          %63 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.fill ins(%cst_0 : f16) outs(%62 : memref<64x32xf16>)
          linalg.matmul ins(%41, %60 : memref<64x64xf16>, memref<64x32xf16>) outs(%62 : memref<64x32xf16>)
          loom.semaphore_give %60 : memref<64x32xf16>
          loom.semaphore_give %41 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%62, %25 : memref<64x32xf16>, memref<64xf16>) outs(%63 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_7: f16, %out: f16):
            %88 = arith.mulf %in_7, %cst_1 : f16
            %89 = math.powf %cst, %88 : f16
            %90 = arith.mulf %in, %89 : f16
            linalg.yield %90 : f16
          }
          loom.semaphore_give %62 : memref<64x32xf16>
          %64 = arith.addi %21, %c1 : index
          %65 = arith.muli %64, %c64 : index
          %66 = arith.ceildivui %65, %c64 : index
          %67 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %68 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          %69 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          %70 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %71 = loom.semaphore_take %70 : memref<64x64xf16> -> memref<64x64xf16>
          %72 = loom.alloc [64] on @L1 : memref<64xf16>
          %73 = loom.semaphore_take %72 : memref<64xf16> -> memref<64xf16>
          %74 = loom.alloc [64] on @L1 : memref<64xf16>
          %75 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
          %76 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %77 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %66 step %c1 {
            %88 = arith.muli %arg14, %c64 : index
            %89 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %90 = arith.muli %arg12, %c262144 : index
            %91 = arith.addi %89, %90 : index
            %92 = arith.muli %39, %c65536 overflow<nsw> : index
            %93 = arith.addi %91, %92 : index
            %94 = arith.muli %21, %c16384 : index
            %95 = arith.addi %93, %94 : index
            %96 = arith.addi %95, %88 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%96], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_7, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %97 = arith.addi %30, %88 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%97], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%46, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%97], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_9, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%46, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%71, %25, %73, %75 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%71 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
              %104 = arith.mulf %in_12, %cst_1 : f16
              %105 = arith.mulf %in_11, %cst_1 : f16
              %106 = arith.subf %105, %104 : f16
              %107 = math.powf %cst, %106 : f16
              %108 = arith.mulf %in, %107 : f16
              %109 = arith.mulf %108, %in_13 : f16
              linalg.yield %109 : f16
            }
            loom.semaphore_give %75 : memref<64xf16>
            loom.semaphore_give %73 : memref<64xf16>
            %98 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %99 = arith.muli %arg12, %c4194304 : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.muli %arg8, %c2048 : index
            %102 = arith.addi %100, %101 : index
            %103 = arith.addi %102, %47 : index
            %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_10, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %58], LR : [%46, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.fill ins(%cst_0 : f16) outs(%69 : memref<64x32xf16>)
            linalg.matmul ins(%71, %77 : memref<64x64xf16>, memref<64x32xf16>) outs(%69 : memref<64x32xf16>)
            loom.semaphore_give %77 : memref<64x32xf16>
            loom.semaphore_give %71 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %69 : memref<64x32xf16>, memref<64x32xf16>) outs(%63 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_11: f16, %out: f16):
              %104 = arith.addf %in, %in_11 : f16
              linalg.yield %104 : f16
            }
            loom.semaphore_give %69 : memref<64x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %78 = loom.alloc [1] on @L1 : memref<f16>
          %79 = loom.semaphore_take %78 : memref<f16> -> memref<f16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %80 = arith.addi %35, %c3 : index
          loom.copy %reinterpret_cast_4, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %35], LR : [%c7, %80]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %81 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %82 = arith.muli %arg12, %c4194304 : index
          %83 = arith.addi %81, %82 : index
          %84 = arith.muli %arg8, %c2048 : index
          %85 = arith.addi %83, %84 : index
          %86 = arith.addi %85, %47 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_5, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %58], LR : [%46, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %68, %79 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%68 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
            %88 = arith.mulf %in_7, %in_8 : f16
            %89 = arith.addf %in, %88 : f16
            linalg.yield %89 : f16
          }
          loom.semaphore_give %79 : memref<f16>
          loom.semaphore_give %63 : memref<64x32xf16>
          %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %87 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%68 : memref<64x32xf16>) outs(%87 : memref<64x32xf16>)
          loom.copy %87, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%32, %58], LR : [%46, %58]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %87 : memref<64x32xf16>
          loom.semaphore_give %68 : memref<64x32xf16>
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
      %cst = arith.constant 2.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
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
        %37 = arith.divui %20, %c64 : index
        %38 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %39 = loom.semaphore_take %38 : memref<64x64xf16> -> memref<64x64xf16>
        %40 = arith.muli %arg12, %c65536 : index
        %41 = arith.addi %24, %40 : index
        %42 = arith.muli %37, %c64 overflow<nsw> : index
        %43 = arith.addi %41, %42 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
        %44 = arith.addi %30, %c3 : index
        loom.copy %reinterpret_cast_2, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%44, %36]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
        %45 = arith.muli %arg10, %c32 : index
        %46 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
        %47 = loom.semaphore_take %46 : memref<32x64xf16> -> memref<32x64xf16>
        %48 = arith.muli %arg11, %c2097152 overflow<nsw> : index
        %49 = arith.muli %arg12, %c1048576 : index
        %50 = arith.addi %48, %49 : index
        %51 = arith.muli %arg8, %c131072 : index
        %52 = arith.addi %50, %51 : index
        %53 = arith.muli %arg10, %c2048 : index
        %54 = arith.addi %52, %53 : index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %55 = arith.addi %arg10, %32 : index
        %56 = arith.addi %55, %33 : index
        loom.copy %reinterpret_cast_3, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %56], LR : [%44, %56]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %57 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %58 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
        linalg.transpose ins(%47 : memref<32x64xf16>) outs(%58 : memref<64x32xf16>) permutation = [1, 0] 
        loom.semaphore_give %47 : memref<32x64xf16>
        %59 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %60 = loom.semaphore_take %59 : memref<64x32xf16> -> memref<64x32xf16>
        %61 = loom.semaphore_take %59 : memref<64x32xf16> -> memref<64x32xf16>
        linalg.fill ins(%cst_0 : f16) outs(%60 : memref<64x32xf16>)
        linalg.matmul ins(%39, %58 : memref<64x64xf16>, memref<64x32xf16>) outs(%60 : memref<64x32xf16>)
        loom.semaphore_give %58 : memref<64x32xf16>
        loom.semaphore_give %39 : memref<64x64xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%60, %23 : memref<64x32xf16>, memref<64xf16>) outs(%61 : memref<64x32xf16>) {
        ^bb0(%in: f16, %in_7: f16, %out: f16):
          %86 = arith.mulf %in_7, %cst_1 : f16
          %87 = math.powf %cst, %86 : f16
          %88 = arith.mulf %in, %87 : f16
          linalg.yield %88 : f16
        }
        loom.semaphore_give %60 : memref<64x32xf16>
        %62 = arith.addi %arg9, %c1 : index
        %63 = arith.muli %62, %c64 : index
        %64 = arith.ceildivui %63, %c64 : index
        %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %66 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
        %67 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
        %68 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %69 = loom.semaphore_take %68 : memref<64x64xf16> -> memref<64x64xf16>
        %70 = loom.alloc [64] on @L1 : memref<64xf16>
        %71 = loom.semaphore_take %70 : memref<64xf16> -> memref<64xf16>
        %72 = loom.alloc [64] on @L1 : memref<64xf16>
        %73 = loom.semaphore_take %72 : memref<64xf16> -> memref<64xf16>
        %74 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %75 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
        scf.for %arg13 = %c0 to %64 step %c1 {
          %86 = arith.muli %arg13, %c64 : index
          %87 = arith.muli %arg11, %c524288 overflow<nsw> : index
          %88 = arith.muli %arg12, %c262144 : index
          %89 = arith.addi %87, %88 : index
          %90 = arith.muli %37, %c65536 overflow<nsw> : index
          %91 = arith.addi %89, %90 : index
          %92 = arith.muli %arg9, %c16384 : index
          %93 = arith.addi %91, %92 : index
          %94 = arith.addi %93, %86 : index
          %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%94], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_7, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
          %95 = arith.addi %28, %86 : index
          %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%95], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_8, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%44, %36]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%95], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_9, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%44, %36]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%69, %23, %71, %73 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%69 : memref<64x64xf16>) {
          ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
            %102 = arith.mulf %in_12, %cst_1 : f16
            %103 = arith.mulf %in_11, %cst_1 : f16
            %104 = arith.subf %103, %102 : f16
            %105 = math.powf %cst, %104 : f16
            %106 = arith.mulf %in, %105 : f16
            %107 = arith.mulf %106, %in_13 : f16
            linalg.yield %107 : f16
          }
          loom.semaphore_give %73 : memref<64xf16>
          loom.semaphore_give %71 : memref<64xf16>
          %96 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %97 = arith.muli %arg12, %c4194304 : index
          %98 = arith.addi %96, %97 : index
          %99 = arith.muli %arg8, %c2048 : index
          %100 = arith.addi %98, %99 : index
          %101 = arith.addi %100, %45 : index
          %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%101], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_10, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %56], LR : [%44, %56]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.fill ins(%cst_0 : f16) outs(%67 : memref<64x32xf16>)
          linalg.matmul ins(%69, %75 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf16>)
          loom.semaphore_give %75 : memref<64x32xf16>
          loom.semaphore_give %69 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%61, %67 : memref<64x32xf16>, memref<64x32xf16>) outs(%61 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_11: f16, %out: f16):
            %102 = arith.addf %in, %in_11 : f16
            linalg.yield %102 : f16
          }
          loom.semaphore_give %67 : memref<64x32xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        loom.semaphore_give %23 : memref<64xf16>
        %76 = loom.alloc [1] on @L1 : memref<f16>
        %77 = loom.semaphore_take %76 : memref<f16> -> memref<f16>
        %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%20], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
        %78 = arith.addi %33, %c3 : index
        loom.copy %reinterpret_cast_4, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %33], LR : [%c7, %78]) : memref<f16, strided<[], offset: ?>> to memref<f16>
        %79 = arith.muli %arg11, %c8388608 overflow<nsw> : index
        %80 = arith.muli %arg12, %c4194304 : index
        %81 = arith.addi %79, %80 : index
        %82 = arith.muli %arg8, %c2048 : index
        %83 = arith.addi %81, %82 : index
        %84 = arith.addi %83, %45 : index
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%84], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_5, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %56], LR : [%44, %56]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%61, %66, %77 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%66 : memref<64x32xf16>) {
        ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
          %86 = arith.mulf %in_7, %in_8 : f16
          %87 = arith.addf %in, %86 : f16
          linalg.yield %87 : f16
        }
        loom.semaphore_give %77 : memref<f16>
        loom.semaphore_give %61 : memref<64x32xf16>
        %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%84], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
        %85 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
        loom.sync ins(%66 : memref<64x32xf16>) outs(%85 : memref<64x32xf16>)
        loom.copy %85, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%30, %56], LR : [%44, %56]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %85 : memref<64x32xf16>
        loom.semaphore_give %66 : memref<64x32xf16>
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
      %cst = arith.constant 2.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
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
        %37 = arith.divui %20, %c64 : index
        %38 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %39 = loom.semaphore_take %38 : memref<64x64xf16> -> memref<64x64xf16>
        %40 = arith.muli %arg12, %c65536 : index
        %41 = arith.addi %24, %40 : index
        %42 = arith.muli %37, %c64 overflow<nsw> : index
        %43 = arith.addi %41, %42 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
        %44 = arith.addi %30, %c3 : index
        loom.copy %reinterpret_cast_2, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%44, %36]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
        %45 = arith.muli %arg10, %c32 : index
        %46 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
        %47 = loom.semaphore_take %46 : memref<32x64xf16> -> memref<32x64xf16>
        %48 = arith.muli %arg11, %c2097152 overflow<nsw> : index
        %49 = arith.muli %arg12, %c1048576 : index
        %50 = arith.addi %48, %49 : index
        %51 = arith.muli %arg8, %c131072 : index
        %52 = arith.addi %50, %51 : index
        %53 = arith.muli %arg10, %c2048 : index
        %54 = arith.addi %52, %53 : index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %55 = arith.addi %arg10, %32 : index
        %56 = arith.addi %55, %33 : index
        loom.copy %reinterpret_cast_3, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %56], LR : [%44, %56]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %57 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %58 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
        linalg.transpose ins(%47 : memref<32x64xf16>) outs(%58 : memref<64x32xf16>) permutation = [1, 0] 
        loom.semaphore_give %47 : memref<32x64xf16>
        %59 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %60 = loom.semaphore_take %59 : memref<64x32xf16> -> memref<64x32xf16>
        %61 = loom.semaphore_take %59 : memref<64x32xf16> -> memref<64x32xf16>
        linalg.fill ins(%cst_0 : f16) outs(%60 : memref<64x32xf16>)
        linalg.matmul ins(%39, %58 : memref<64x64xf16>, memref<64x32xf16>) outs(%60 : memref<64x32xf16>)
        loom.semaphore_give %58 : memref<64x32xf16>
        loom.semaphore_give %39 : memref<64x64xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%60, %23 : memref<64x32xf16>, memref<64xf16>) outs(%61 : memref<64x32xf16>) {
        ^bb0(%in: f16, %in_7: f16, %out: f16):
          %86 = arith.mulf %in_7, %cst_1 : f16
          %87 = math.powf %cst, %86 : f16
          %88 = arith.mulf %in, %87 : f16
          linalg.yield %88 : f16
        }
        loom.semaphore_give %60 : memref<64x32xf16>
        %62 = arith.addi %arg9, %c1 : index
        %63 = arith.muli %62, %c64 : index
        %64 = arith.ceildivui %63, %c64 : index
        %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %66 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
        %67 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
        %68 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %69 = loom.semaphore_take %68 : memref<64x64xf16> -> memref<64x64xf16>
        %70 = loom.alloc [64] on @L1 : memref<64xf16>
        %71 = loom.semaphore_take %70 : memref<64xf16> -> memref<64xf16>
        %72 = loom.alloc [64] on @L1 : memref<64xf16>
        %73 = loom.semaphore_take %72 : memref<64xf16> -> memref<64xf16>
        %74 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %75 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
        scf.for %arg13 = %c0 to %64 step %c1 {
          %86 = arith.muli %arg13, %c64 : index
          %87 = arith.muli %arg11, %c524288 overflow<nsw> : index
          %88 = arith.muli %arg12, %c262144 : index
          %89 = arith.addi %87, %88 : index
          %90 = arith.muli %37, %c65536 overflow<nsw> : index
          %91 = arith.addi %89, %90 : index
          %92 = arith.muli %arg9, %c16384 : index
          %93 = arith.addi %91, %92 : index
          %94 = arith.addi %93, %86 : index
          %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%94], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_7, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
          %95 = arith.addi %28, %86 : index
          %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%95], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_8, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%44, %36]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%95], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_9, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%44, %36]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%69, %23, %71, %73 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%69 : memref<64x64xf16>) {
          ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
            %102 = arith.mulf %in_12, %cst_1 : f16
            %103 = arith.mulf %in_11, %cst_1 : f16
            %104 = arith.subf %103, %102 : f16
            %105 = math.powf %cst, %104 : f16
            %106 = arith.mulf %in, %105 : f16
            %107 = arith.mulf %106, %in_13 : f16
            linalg.yield %107 : f16
          }
          loom.semaphore_give %73 : memref<64xf16>
          loom.semaphore_give %71 : memref<64xf16>
          %96 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %97 = arith.muli %arg12, %c4194304 : index
          %98 = arith.addi %96, %97 : index
          %99 = arith.muli %arg8, %c2048 : index
          %100 = arith.addi %98, %99 : index
          %101 = arith.addi %100, %45 : index
          %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%101], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_10, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %56], LR : [%44, %56]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.fill ins(%cst_0 : f16) outs(%67 : memref<64x32xf16>)
          linalg.matmul ins(%69, %75 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf16>)
          loom.semaphore_give %75 : memref<64x32xf16>
          loom.semaphore_give %69 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%61, %67 : memref<64x32xf16>, memref<64x32xf16>) outs(%61 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_11: f16, %out: f16):
            %102 = arith.addf %in, %in_11 : f16
            linalg.yield %102 : f16
          }
          loom.semaphore_give %67 : memref<64x32xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        loom.semaphore_give %23 : memref<64xf16>
        %76 = loom.alloc [1] on @L1 : memref<f16>
        %77 = loom.semaphore_take %76 : memref<f16> -> memref<f16>
        %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%20], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
        %78 = arith.addi %33, %c3 : index
        loom.copy %reinterpret_cast_4, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %33], LR : [%c7, %78]) : memref<f16, strided<[], offset: ?>> to memref<f16>
        %79 = arith.muli %arg11, %c8388608 overflow<nsw> : index
        %80 = arith.muli %arg12, %c4194304 : index
        %81 = arith.addi %79, %80 : index
        %82 = arith.muli %arg8, %c2048 : index
        %83 = arith.addi %81, %82 : index
        %84 = arith.addi %83, %45 : index
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%84], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_5, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %56], LR : [%44, %56]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%61, %66, %77 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%66 : memref<64x32xf16>) {
        ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
          %86 = arith.mulf %in_7, %in_8 : f16
          %87 = arith.addf %in, %86 : f16
          linalg.yield %87 : f16
        }
        loom.semaphore_give %77 : memref<f16>
        loom.semaphore_give %61 : memref<64x32xf16>
        %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%84], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
        %85 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
        loom.sync ins(%66 : memref<64x32xf16>) outs(%85 : memref<64x32xf16>)
        loom.copy %85, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%30, %56], LR : [%44, %56]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %85 : memref<64x32xf16>
        loom.semaphore_give %66 : memref<64x32xf16>
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
      %cst = arith.constant 2.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
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
          %38 = arith.divui %22, %c64 : index
          %39 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %40 = loom.semaphore_take %39 : memref<64x64xf16> -> memref<64x64xf16>
          %41 = arith.muli %arg12, %c65536 : index
          %42 = arith.addi %26, %41 : index
          %43 = arith.muli %38, %c64 overflow<nsw> : index
          %44 = arith.addi %42, %43 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          %45 = arith.addi %32, %34 : index
          %46 = arith.addi %32, %c1 : index
          %47 = arith.addi %46, %34 : index
          loom.copy %reinterpret_cast_2, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%45, %36], LR : [%47, %37]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          %48 = arith.muli %arg10, %c32 : index
          %49 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %50 = loom.semaphore_take %49 : memref<32x64xf16> -> memref<32x64xf16>
          %51 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %52 = arith.muli %arg12, %c1048576 : index
          %53 = arith.addi %51, %52 : index
          %54 = arith.muli %arg8, %c131072 : index
          %55 = arith.addi %53, %54 : index
          %56 = arith.muli %arg10, %c2048 : index
          %57 = arith.addi %55, %56 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%57], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          %58 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_3, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %59 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %60 = loom.semaphore_take %59 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.transpose ins(%50 : memref<32x64xf16>) outs(%60 : memref<64x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %50 : memref<32x64xf16>
          %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          %63 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.fill ins(%cst_0 : f16) outs(%62 : memref<64x32xf16>)
          linalg.matmul ins(%40, %60 : memref<64x64xf16>, memref<64x32xf16>) outs(%62 : memref<64x32xf16>)
          loom.semaphore_give %60 : memref<64x32xf16>
          loom.semaphore_give %40 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%62, %25 : memref<64x32xf16>, memref<64xf16>) outs(%63 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_7: f16, %out: f16):
            %88 = arith.mulf %in_7, %cst_1 : f16
            %89 = math.powf %cst, %88 : f16
            %90 = arith.mulf %in, %89 : f16
            linalg.yield %90 : f16
          }
          loom.semaphore_give %62 : memref<64x32xf16>
          %64 = arith.addi %21, %c1 : index
          %65 = arith.muli %64, %c64 : index
          %66 = arith.ceildivui %65, %c64 : index
          %67 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %68 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          %69 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          %70 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %71 = loom.semaphore_take %70 : memref<64x64xf16> -> memref<64x64xf16>
          %72 = loom.alloc [64] on @L1 : memref<64xf16>
          %73 = loom.semaphore_take %72 : memref<64xf16> -> memref<64xf16>
          %74 = loom.alloc [64] on @L1 : memref<64xf16>
          %75 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
          %76 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %77 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %66 step %c1 {
            %88 = arith.muli %arg14, %c64 : index
            %89 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %90 = arith.muli %arg12, %c262144 : index
            %91 = arith.addi %89, %90 : index
            %92 = arith.muli %38, %c65536 overflow<nsw> : index
            %93 = arith.addi %91, %92 : index
            %94 = arith.muli %21, %c16384 : index
            %95 = arith.addi %93, %94 : index
            %96 = arith.addi %95, %88 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%96], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_7, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %36], LR : [%35, %37]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %97 = arith.addi %30, %88 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%97], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%45, %36], LR : [%47, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%97], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_9, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%45, %36], LR : [%47, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%71, %25, %73, %75 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%71 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
              %104 = arith.mulf %in_12, %cst_1 : f16
              %105 = arith.mulf %in_11, %cst_1 : f16
              %106 = arith.subf %105, %104 : f16
              %107 = math.powf %cst, %106 : f16
              %108 = arith.mulf %in, %107 : f16
              %109 = arith.mulf %108, %in_13 : f16
              linalg.yield %109 : f16
            }
            loom.semaphore_give %75 : memref<64xf16>
            loom.semaphore_give %73 : memref<64xf16>
            %98 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %99 = arith.muli %arg12, %c4194304 : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.muli %arg8, %c2048 : index
            %102 = arith.addi %100, %101 : index
            %103 = arith.addi %102, %48 : index
            %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_10, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.fill ins(%cst_0 : f16) outs(%69 : memref<64x32xf16>)
            linalg.matmul ins(%71, %77 : memref<64x64xf16>, memref<64x32xf16>) outs(%69 : memref<64x32xf16>)
            loom.semaphore_give %77 : memref<64x32xf16>
            loom.semaphore_give %71 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %69 : memref<64x32xf16>, memref<64x32xf16>) outs(%63 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_11: f16, %out: f16):
              %104 = arith.addf %in, %in_11 : f16
              linalg.yield %104 : f16
            }
            loom.semaphore_give %69 : memref<64x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %78 = loom.alloc [1] on @L1 : memref<f16>
          %79 = loom.semaphore_take %78 : memref<f16> -> memref<f16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %80 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_4, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%80, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %81 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %82 = arith.muli %arg12, %c4194304 : index
          %83 = arith.addi %81, %82 : index
          %84 = arith.muli %arg8, %c2048 : index
          %85 = arith.addi %83, %84 : index
          %86 = arith.addi %85, %48 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_5, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %68, %79 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%68 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
            %88 = arith.mulf %in_7, %in_8 : f16
            %89 = arith.addf %in, %88 : f16
            linalg.yield %89 : f16
          }
          loom.semaphore_give %79 : memref<f16>
          loom.semaphore_give %63 : memref<64x32xf16>
          %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %87 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%68 : memref<64x32xf16>) outs(%87 : memref<64x32xf16>)
          loom.copy %87, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %87 : memref<64x32xf16>
          loom.semaphore_give %68 : memref<64x32xf16>
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
      %cst = arith.constant 2.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
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
          %38 = arith.divui %22, %c64 : index
          %39 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %40 = loom.semaphore_take %39 : memref<64x64xf16> -> memref<64x64xf16>
          %41 = arith.muli %arg12, %c65536 : index
          %42 = arith.addi %26, %41 : index
          %43 = arith.muli %38, %c64 overflow<nsw> : index
          %44 = arith.addi %42, %43 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          %45 = arith.addi %32, %34 : index
          %46 = arith.addi %32, %c1 : index
          %47 = arith.addi %46, %34 : index
          loom.copy %reinterpret_cast_2, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%45, %36], LR : [%47, %37]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          %48 = arith.muli %arg10, %c32 : index
          %49 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %50 = loom.semaphore_take %49 : memref<32x64xf16> -> memref<32x64xf16>
          %51 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %52 = arith.muli %arg12, %c1048576 : index
          %53 = arith.addi %51, %52 : index
          %54 = arith.muli %arg8, %c131072 : index
          %55 = arith.addi %53, %54 : index
          %56 = arith.muli %arg10, %c2048 : index
          %57 = arith.addi %55, %56 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%57], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          %58 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_3, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %59 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %60 = loom.semaphore_take %59 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.transpose ins(%50 : memref<32x64xf16>) outs(%60 : memref<64x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %50 : memref<32x64xf16>
          %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          %63 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.fill ins(%cst_0 : f16) outs(%62 : memref<64x32xf16>)
          linalg.matmul ins(%40, %60 : memref<64x64xf16>, memref<64x32xf16>) outs(%62 : memref<64x32xf16>)
          loom.semaphore_give %60 : memref<64x32xf16>
          loom.semaphore_give %40 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%62, %25 : memref<64x32xf16>, memref<64xf16>) outs(%63 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_7: f16, %out: f16):
            %88 = arith.mulf %in_7, %cst_1 : f16
            %89 = math.powf %cst, %88 : f16
            %90 = arith.mulf %in, %89 : f16
            linalg.yield %90 : f16
          }
          loom.semaphore_give %62 : memref<64x32xf16>
          %64 = arith.addi %21, %c1 : index
          %65 = arith.muli %64, %c64 : index
          %66 = arith.ceildivui %65, %c64 : index
          %67 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %68 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          %69 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          %70 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %71 = loom.semaphore_take %70 : memref<64x64xf16> -> memref<64x64xf16>
          %72 = loom.alloc [64] on @L1 : memref<64xf16>
          %73 = loom.semaphore_take %72 : memref<64xf16> -> memref<64xf16>
          %74 = loom.alloc [64] on @L1 : memref<64xf16>
          %75 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
          %76 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %77 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %66 step %c1 {
            %88 = arith.muli %arg14, %c64 : index
            %89 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %90 = arith.muli %arg12, %c262144 : index
            %91 = arith.addi %89, %90 : index
            %92 = arith.muli %38, %c65536 overflow<nsw> : index
            %93 = arith.addi %91, %92 : index
            %94 = arith.muli %21, %c16384 : index
            %95 = arith.addi %93, %94 : index
            %96 = arith.addi %95, %88 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%96], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_7, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %36], LR : [%35, %37]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %97 = arith.addi %30, %88 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%97], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%45, %36], LR : [%47, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%97], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_9, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%45, %36], LR : [%47, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%71, %25, %73, %75 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%71 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
              %104 = arith.mulf %in_12, %cst_1 : f16
              %105 = arith.mulf %in_11, %cst_1 : f16
              %106 = arith.subf %105, %104 : f16
              %107 = math.powf %cst, %106 : f16
              %108 = arith.mulf %in, %107 : f16
              %109 = arith.mulf %108, %in_13 : f16
              linalg.yield %109 : f16
            }
            loom.semaphore_give %75 : memref<64xf16>
            loom.semaphore_give %73 : memref<64xf16>
            %98 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %99 = arith.muli %arg12, %c4194304 : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.muli %arg8, %c2048 : index
            %102 = arith.addi %100, %101 : index
            %103 = arith.addi %102, %48 : index
            %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_10, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.fill ins(%cst_0 : f16) outs(%69 : memref<64x32xf16>)
            linalg.matmul ins(%71, %77 : memref<64x64xf16>, memref<64x32xf16>) outs(%69 : memref<64x32xf16>)
            loom.semaphore_give %77 : memref<64x32xf16>
            loom.semaphore_give %71 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %69 : memref<64x32xf16>, memref<64x32xf16>) outs(%63 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_11: f16, %out: f16):
              %104 = arith.addf %in, %in_11 : f16
              linalg.yield %104 : f16
            }
            loom.semaphore_give %69 : memref<64x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %78 = loom.alloc [1] on @L1 : memref<f16>
          %79 = loom.semaphore_take %78 : memref<f16> -> memref<f16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %80 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_4, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%80, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %81 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %82 = arith.muli %arg12, %c4194304 : index
          %83 = arith.addi %81, %82 : index
          %84 = arith.muli %arg8, %c2048 : index
          %85 = arith.addi %83, %84 : index
          %86 = arith.addi %85, %48 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_5, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %68, %79 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%68 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
            %88 = arith.mulf %in_7, %in_8 : f16
            %89 = arith.addf %in, %88 : f16
            linalg.yield %89 : f16
          }
          loom.semaphore_give %79 : memref<f16>
          loom.semaphore_give %63 : memref<64x32xf16>
          %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %87 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%68 : memref<64x32xf16>) outs(%87 : memref<64x32xf16>)
          loom.copy %87, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %87 : memref<64x32xf16>
          loom.semaphore_give %68 : memref<64x32xf16>
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
      %cst = arith.constant 2.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
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
          %38 = arith.divui %22, %c64 : index
          %39 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %40 = loom.semaphore_take %39 : memref<64x64xf16> -> memref<64x64xf16>
          %41 = arith.muli %arg12, %c65536 : index
          %42 = arith.addi %26, %41 : index
          %43 = arith.muli %38, %c64 overflow<nsw> : index
          %44 = arith.addi %42, %43 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          %45 = arith.addi %32, %34 : index
          %46 = arith.addi %32, %c1 : index
          %47 = arith.addi %46, %34 : index
          loom.copy %reinterpret_cast_2, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%45, %36], LR : [%47, %37]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          %48 = arith.muli %arg10, %c32 : index
          %49 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %50 = loom.semaphore_take %49 : memref<32x64xf16> -> memref<32x64xf16>
          %51 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %52 = arith.muli %arg12, %c1048576 : index
          %53 = arith.addi %51, %52 : index
          %54 = arith.muli %arg8, %c131072 : index
          %55 = arith.addi %53, %54 : index
          %56 = arith.muli %arg10, %c2048 : index
          %57 = arith.addi %55, %56 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%57], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          %58 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_3, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %59 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %60 = loom.semaphore_take %59 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.transpose ins(%50 : memref<32x64xf16>) outs(%60 : memref<64x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %50 : memref<32x64xf16>
          %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          %63 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.fill ins(%cst_0 : f16) outs(%62 : memref<64x32xf16>)
          linalg.matmul ins(%40, %60 : memref<64x64xf16>, memref<64x32xf16>) outs(%62 : memref<64x32xf16>)
          loom.semaphore_give %60 : memref<64x32xf16>
          loom.semaphore_give %40 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%62, %25 : memref<64x32xf16>, memref<64xf16>) outs(%63 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_7: f16, %out: f16):
            %88 = arith.mulf %in_7, %cst_1 : f16
            %89 = math.powf %cst, %88 : f16
            %90 = arith.mulf %in, %89 : f16
            linalg.yield %90 : f16
          }
          loom.semaphore_give %62 : memref<64x32xf16>
          %64 = arith.addi %21, %c1 : index
          %65 = arith.muli %64, %c64 : index
          %66 = arith.ceildivui %65, %c64 : index
          %67 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %68 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          %69 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          %70 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %71 = loom.semaphore_take %70 : memref<64x64xf16> -> memref<64x64xf16>
          %72 = loom.alloc [64] on @L1 : memref<64xf16>
          %73 = loom.semaphore_take %72 : memref<64xf16> -> memref<64xf16>
          %74 = loom.alloc [64] on @L1 : memref<64xf16>
          %75 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
          %76 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %77 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %66 step %c1 {
            %88 = arith.muli %arg14, %c64 : index
            %89 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %90 = arith.muli %arg12, %c262144 : index
            %91 = arith.addi %89, %90 : index
            %92 = arith.muli %38, %c65536 overflow<nsw> : index
            %93 = arith.addi %91, %92 : index
            %94 = arith.muli %21, %c16384 : index
            %95 = arith.addi %93, %94 : index
            %96 = arith.addi %95, %88 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%96], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_7, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%35, %36], LR : [%35, %37]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %97 = arith.addi %30, %88 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%97], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%45, %36], LR : [%47, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%97], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_9, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%45, %36], LR : [%47, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%71, %25, %73, %75 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%71 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
              %104 = arith.mulf %in_12, %cst_1 : f16
              %105 = arith.mulf %in_11, %cst_1 : f16
              %106 = arith.subf %105, %104 : f16
              %107 = math.powf %cst, %106 : f16
              %108 = arith.mulf %in, %107 : f16
              %109 = arith.mulf %108, %in_13 : f16
              linalg.yield %109 : f16
            }
            loom.semaphore_give %75 : memref<64xf16>
            loom.semaphore_give %73 : memref<64xf16>
            %98 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %99 = arith.muli %arg12, %c4194304 : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.muli %arg8, %c2048 : index
            %102 = arith.addi %100, %101 : index
            %103 = arith.addi %102, %48 : index
            %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_10, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.fill ins(%cst_0 : f16) outs(%69 : memref<64x32xf16>)
            linalg.matmul ins(%71, %77 : memref<64x64xf16>, memref<64x32xf16>) outs(%69 : memref<64x32xf16>)
            loom.semaphore_give %77 : memref<64x32xf16>
            loom.semaphore_give %71 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %69 : memref<64x32xf16>, memref<64x32xf16>) outs(%63 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_11: f16, %out: f16):
              %104 = arith.addf %in, %in_11 : f16
              linalg.yield %104 : f16
            }
            loom.semaphore_give %69 : memref<64x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %78 = loom.alloc [1] on @L1 : memref<f16>
          %79 = loom.semaphore_take %78 : memref<f16> -> memref<f16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %80 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_4, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%80, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %81 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %82 = arith.muli %arg12, %c4194304 : index
          %83 = arith.addi %81, %82 : index
          %84 = arith.muli %arg8, %c2048 : index
          %85 = arith.addi %83, %84 : index
          %86 = arith.addi %85, %48 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_5, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %68, %79 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%68 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
            %88 = arith.mulf %in_7, %in_8 : f16
            %89 = arith.addf %in, %88 : f16
            linalg.yield %89 : f16
          }
          loom.semaphore_give %79 : memref<f16>
          loom.semaphore_give %63 : memref<64x32xf16>
          %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %87 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%68 : memref<64x32xf16>) outs(%87 : memref<64x32xf16>)
          loom.copy %87, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %87 : memref<64x32xf16>
          loom.semaphore_give %68 : memref<64x32xf16>
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
      %cst = arith.constant 2.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
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
          %38 = arith.divui %22, %c64 : index
          %39 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %40 = loom.semaphore_take %39 : memref<64x64xf16> -> memref<64x64xf16>
          %41 = arith.muli %arg12, %c65536 : index
          %42 = arith.addi %26, %41 : index
          %43 = arith.muli %38, %c64 overflow<nsw> : index
          %44 = arith.addi %42, %43 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          %45 = arith.addi %32, %34 : index
          %46 = arith.addi %32, %c1 : index
          %47 = arith.addi %46, %34 : index
          loom.copy %reinterpret_cast_2, %40 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%45, %36], LR : [%47, %37]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          %48 = arith.muli %arg10, %c32 : index
          %49 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %50 = loom.semaphore_take %49 : memref<32x64xf16> -> memref<32x64xf16>
          %51 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %52 = arith.muli %arg12, %c1048576 : index
          %53 = arith.addi %51, %52 : index
          %54 = arith.muli %arg8, %c131072 : index
          %55 = arith.addi %53, %54 : index
          %56 = arith.muli %arg10, %c2048 : index
          %57 = arith.addi %55, %56 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%57], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
          %58 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_3, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
          %59 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %60 = loom.semaphore_take %59 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.transpose ins(%50 : memref<32x64xf16>) outs(%60 : memref<64x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %50 : memref<32x64xf16>
          %61 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %62 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          %63 = loom.semaphore_take %61 : memref<64x32xf16> -> memref<64x32xf16>
          linalg.fill ins(%cst_0 : f16) outs(%62 : memref<64x32xf16>)
          linalg.matmul ins(%40, %60 : memref<64x64xf16>, memref<64x32xf16>) outs(%62 : memref<64x32xf16>)
          loom.semaphore_give %60 : memref<64x32xf16>
          loom.semaphore_give %40 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%62, %25 : memref<64x32xf16>, memref<64xf16>) outs(%63 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_7: f16, %out: f16):
            %88 = arith.mulf %in_7, %cst_1 : f16
            %89 = math.powf %cst, %88 : f16
            %90 = arith.mulf %in, %89 : f16
            linalg.yield %90 : f16
          }
          loom.semaphore_give %62 : memref<64x32xf16>
          %64 = arith.addi %21, %c1 : index
          %65 = arith.muli %64, %c64 : index
          %66 = arith.ceildivui %65, %c64 : index
          %67 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %68 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          %69 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          %70 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %71 = loom.semaphore_take %70 : memref<64x64xf16> -> memref<64x64xf16>
          %72 = loom.alloc [64] on @L1 : memref<64xf16>
          %73 = loom.semaphore_take %72 : memref<64xf16> -> memref<64xf16>
          %74 = loom.alloc [64] on @L1 : memref<64xf16>
          %75 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
          %76 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %77 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %66 step %c1 {
            %88 = arith.muli %arg14, %c64 : index
            %89 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %90 = arith.muli %arg12, %c262144 : index
            %91 = arith.addi %89, %90 : index
            %92 = arith.muli %38, %c65536 overflow<nsw> : index
            %93 = arith.addi %91, %92 : index
            %94 = arith.muli %21, %c16384 : index
            %95 = arith.addi %93, %94 : index
            %96 = arith.addi %95, %88 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%96], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_7, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%35, %36], LR : [%35, %37]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %97 = arith.addi %30, %88 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%97], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%45, %36], LR : [%47, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%97], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_9, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%45, %36], LR : [%47, %37]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%71, %25, %73, %75 : memref<64x64xf16>, memref<64xf16>, memref<64xf16>, memref<64xf16>) outs(%71 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
              %104 = arith.mulf %in_12, %cst_1 : f16
              %105 = arith.mulf %in_11, %cst_1 : f16
              %106 = arith.subf %105, %104 : f16
              %107 = math.powf %cst, %106 : f16
              %108 = arith.mulf %in, %107 : f16
              %109 = arith.mulf %108, %in_13 : f16
              linalg.yield %109 : f16
            }
            loom.semaphore_give %75 : memref<64xf16>
            loom.semaphore_give %73 : memref<64xf16>
            %98 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %99 = arith.muli %arg12, %c4194304 : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.muli %arg8, %c2048 : index
            %102 = arith.addi %100, %101 : index
            %103 = arith.addi %102, %48 : index
            %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_10, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.fill ins(%cst_0 : f16) outs(%69 : memref<64x32xf16>)
            linalg.matmul ins(%71, %77 : memref<64x64xf16>, memref<64x32xf16>) outs(%69 : memref<64x32xf16>)
            loom.semaphore_give %77 : memref<64x32xf16>
            loom.semaphore_give %71 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %69 : memref<64x32xf16>, memref<64x32xf16>) outs(%63 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_11: f16, %out: f16):
              %104 = arith.addf %in, %in_11 : f16
              linalg.yield %104 : f16
            }
            loom.semaphore_give %69 : memref<64x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %78 = loom.alloc [1] on @L1 : memref<f16>
          %79 = loom.semaphore_take %78 : memref<f16> -> memref<f16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %80 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_4, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%80, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %81 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %82 = arith.muli %arg12, %c4194304 : index
          %83 = arith.addi %81, %82 : index
          %84 = arith.muli %arg8, %c2048 : index
          %85 = arith.addi %83, %84 : index
          %86 = arith.addi %85, %48 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_5, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %68, %79 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%68 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
            %88 = arith.mulf %in_7, %in_8 : f16
            %89 = arith.addf %in, %88 : f16
            linalg.yield %89 : f16
          }
          loom.semaphore_give %79 : memref<f16>
          loom.semaphore_give %63 : memref<64x32xf16>
          %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %87 = loom.semaphore_take %67 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%68 : memref<64x32xf16>) outs(%87 : memref<64x32xf16>)
          loom.copy %87, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%45, %58], LR : [%47, %58]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %87 : memref<64x32xf16>
          loom.semaphore_give %68 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
}
