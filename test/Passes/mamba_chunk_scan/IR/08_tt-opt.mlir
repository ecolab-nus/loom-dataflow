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
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c4, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          scf.for %arg14 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg13, %c2 overflow<nsw> : index
            %21 = arith.addi %arg8, %20 : index
            %22 = arith.muli %arg14, %c2 overflow<nsw> : index
            %23 = arith.addi %arg9, %22 : index
            %24 = arith.muli %21, %c8 : index
            %25 = arith.muli %23, %c64 : index
            %26 = loom.alloc [64] on @L1 : memref<64xf16>
            %27 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %28 = arith.muli %arg11, %c131072 overflow<nsw> : index
            %29 = arith.muli %21, %c16384 : index
            %30 = arith.addi %28, %29 : index
            %31 = arith.muli %arg12, %c1024 : index
            %32 = arith.addi %30, %31 : index
            %33 = arith.addi %32, %25 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            %34 = arith.muli %arg11, %c2 : index
            %35 = arith.addi %arg9, %34 : index
            %36 = arith.muli %arg12, %c2 : index
            %37 = arith.muli %arg8, %c4 : index
            %38 = arith.addi %36, %37 : index
            %39 = arith.addi %36, %c1 : index
            %40 = arith.addi %39, %37 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %38], LR : [%35, %40]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %41 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %42 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.broadcast ins(%27 : memref<64xf16>) outs(%43 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            %45 = arith.addi %25, %31 : index
            %46 = arith.divui %24, %c64 : index
            %47 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %48 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
            %49 = arith.muli %45, %c64 overflow<nsw> : index
            %50 = arith.addi %28, %49 : index
            %51 = arith.muli %46, %c64 overflow<nsw> : index
            %52 = arith.addi %50, %51 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%52], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %38], LR : [%35, %40]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
            %53 = arith.muli %arg10, %c32 : index
            %54 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %55 = loom.semaphore_take %54 : memref<64x32xf16> -> memref<64x32xf16>
            %56 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %57 = arith.muli %arg12, %c1048576 : index
            %58 = arith.addi %56, %57 : index
            %59 = arith.muli %21, %c32768 : index
            %60 = arith.addi %58, %59 : index
            %61 = arith.addi %60, %53 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%61], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
            %62 = arith.addi %34, %c1 : index
            %63 = arith.addi %arg10, %36 : index
            %64 = arith.addi %63, %37 : index
            loom.copy %reinterpret_cast_1, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%34, %64], LR : [%62, %64]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
            %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %66 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
            %67 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%48, %55 : memref<64x64xf16>, memref<64x32xf16>) outs(%66 : memref<64x32xf16>)
            loom.semaphore_give %55 : memref<64x32xf16>
            loom.semaphore_give %48 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %44 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%67 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %94 = math.exp %in_5 : f16
              %95 = arith.mulf %in, %94 : f16
              linalg.yield %95 : f16
            }
            loom.semaphore_give %66 : memref<64x32xf16>
            loom.semaphore_give %43 : memref<64x32xf16>
            %68 = arith.addi %23, %c1 : index
            %69 = arith.muli %68, %c64 : index
            %70 = arith.ceildivui %69, %c64 : index
            %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
            %73 = loom.alloc [64] on @L1 : memref<64xf16>
            %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
            %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
            %76 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %77 = loom.semaphore_take %76 : memref<32x64xf16> -> memref<32x64xf16>
            %78 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %79 = loom.semaphore_take %78 : memref<32x64xf16> -> memref<32x64xf16>
            %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
            scf.for %arg15 = %c0 to %70 step %c1 {
              %94 = arith.muli %arg15, %c64 : index
              %95 = arith.muli %arg11, %c524288 overflow<nsw> : index
              %96 = arith.muli %arg12, %c262144 : index
              %97 = arith.addi %95, %96 : index
              %98 = arith.muli %46, %c65536 overflow<nsw> : index
              %99 = arith.addi %97, %98 : index
              %100 = arith.muli %23, %c16384 : index
              %101 = arith.addi %99, %100 : index
              %102 = arith.addi %101, %94 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %38], LR : [%35, %40]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
              %103 = arith.addi %32, %94 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_6, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%34, %38], LR : [%62, %40]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              %104 = loom.broadcast ins(%27 : memref<64xf16>) outs(%42 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              %105 = loom.broadcast ins(%75 : memref<64xf16>) outs(%77 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %75 : memref<64xf16>
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_7, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%34, %38], LR : [%62, %40]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              %106 = loom.broadcast ins(%74 : memref<64xf16>) outs(%79 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %74 : memref<64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %104, %105, %106 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%72 : memref<64x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
                %114 = arith.subf %in_9, %in_10 : f16
                %115 = math.exp %114 : f16
                %116 = arith.mulf %in, %115 : f16
                %117 = arith.mulf %116, %in_11 : f16
                linalg.yield %117 : f16
              }
              loom.semaphore_give %79 : memref<32x64xf16>
              loom.semaphore_give %77 : memref<32x64xf16>
              loom.semaphore_give %42 : memref<64x32xf16>
              %107 = arith.addi %94, %31 : index
              %108 = arith.muli %arg11, %c8388608 overflow<nsw> : index
              %109 = arith.muli %107, %c4096 overflow<nsw> : index
              %110 = arith.addi %108, %109 : index
              %111 = arith.muli %21, %c512 : index
              %112 = arith.addi %110, %111 : index
              %113 = arith.addi %112, %53 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%34, %64], LR : [%62, %64]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
              linalg.matmul ins(%72, %81 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf16>)
              loom.semaphore_give %81 : memref<64x32xf16>
              loom.semaphore_give %72 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %27 : memref<64xf16>
            %82 = loom.alloc [1] on @L1 : memref<f16>
            %83 = loom.semaphore_take %82 : memref<f16> -> memref<f16>
            %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
            %84 = arith.addi %37, %c3 : index
            loom.copy %reinterpret_cast_2, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %37], LR : [%c7, %84]) : memref<f16, strided<[], offset: ?>> to memref<f16>
            %85 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %86 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
            %87 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %88 = arith.muli %45, %c4096 overflow<nsw> : index
            %89 = arith.addi %87, %88 : index
            %90 = arith.muli %21, %c512 : index
            %91 = arith.addi %89, %90 : index
            %92 = arith.addi %91, %53 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %64], LR : [%35, %64]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %86, %83 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%86 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %94 = arith.mulf %in_5, %in_6 : f16
              %95 = arith.addf %in, %94 : f16
              linalg.yield %95 : f16
            }
            loom.semaphore_give %83 : memref<f16>
            loom.semaphore_give %67 : memref<64x32xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            %93 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
            loom.sync ins(%86 : memref<64x32xf16>) outs(%93 : memref<64x32xf16>)
            loom.copy %93, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%35, %64], LR : [%35, %64]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %93 : memref<64x32xf16>
            loom.semaphore_give %86 : memref<64x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c2, %c4) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          scf.for %arg14 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg13, %c2 overflow<nsw> : index
            %21 = arith.addi %arg8, %20 : index
            %22 = arith.muli %arg14, %c2 overflow<nsw> : index
            %23 = arith.addi %arg9, %22 : index
            %24 = arith.muli %21, %c8 : index
            %25 = arith.muli %23, %c64 : index
            %26 = loom.alloc [64] on @L1 : memref<64xf16>
            %27 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %28 = arith.muli %arg11, %c131072 overflow<nsw> : index
            %29 = arith.muli %21, %c16384 : index
            %30 = arith.addi %28, %29 : index
            %31 = arith.muli %arg12, %c1024 : index
            %32 = arith.addi %30, %31 : index
            %33 = arith.addi %32, %25 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            %34 = arith.muli %arg12, %c2 : index
            %35 = arith.addi %arg9, %34 : index
            %36 = arith.muli %arg11, %c2 : index
            %37 = arith.muli %arg8, %c4 : index
            %38 = arith.addi %36, %37 : index
            %39 = arith.addi %36, %c1 : index
            %40 = arith.addi %39, %37 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %38], LR : [%35, %40]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %41 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %42 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.broadcast ins(%27 : memref<64xf16>) outs(%43 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            %45 = arith.addi %25, %31 : index
            %46 = arith.divui %24, %c64 : index
            %47 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %48 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
            %49 = arith.muli %45, %c64 overflow<nsw> : index
            %50 = arith.addi %28, %49 : index
            %51 = arith.muli %46, %c64 overflow<nsw> : index
            %52 = arith.addi %50, %51 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%52], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %38], LR : [%35, %40]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
            %53 = arith.muli %arg10, %c32 : index
            %54 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %55 = loom.semaphore_take %54 : memref<64x32xf16> -> memref<64x32xf16>
            %56 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %57 = arith.muli %arg12, %c1048576 : index
            %58 = arith.addi %56, %57 : index
            %59 = arith.muli %21, %c32768 : index
            %60 = arith.addi %58, %59 : index
            %61 = arith.addi %60, %53 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%61], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
            %62 = arith.addi %34, %c1 : index
            %63 = arith.addi %arg10, %36 : index
            %64 = arith.addi %63, %37 : index
            loom.copy %reinterpret_cast_1, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%34, %64], LR : [%62, %64]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
            %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %66 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
            %67 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%48, %55 : memref<64x64xf16>, memref<64x32xf16>) outs(%66 : memref<64x32xf16>)
            loom.semaphore_give %55 : memref<64x32xf16>
            loom.semaphore_give %48 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %44 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%67 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %94 = math.exp %in_5 : f16
              %95 = arith.mulf %in, %94 : f16
              linalg.yield %95 : f16
            }
            loom.semaphore_give %66 : memref<64x32xf16>
            loom.semaphore_give %43 : memref<64x32xf16>
            %68 = arith.addi %23, %c1 : index
            %69 = arith.muli %68, %c64 : index
            %70 = arith.ceildivui %69, %c64 : index
            %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
            %73 = loom.alloc [64] on @L1 : memref<64xf16>
            %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
            %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
            %76 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %77 = loom.semaphore_take %76 : memref<32x64xf16> -> memref<32x64xf16>
            %78 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %79 = loom.semaphore_take %78 : memref<32x64xf16> -> memref<32x64xf16>
            %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
            scf.for %arg15 = %c0 to %70 step %c1 {
              %94 = arith.muli %arg15, %c64 : index
              %95 = arith.muli %arg11, %c524288 overflow<nsw> : index
              %96 = arith.muli %arg12, %c262144 : index
              %97 = arith.addi %95, %96 : index
              %98 = arith.muli %46, %c65536 overflow<nsw> : index
              %99 = arith.addi %97, %98 : index
              %100 = arith.muli %23, %c16384 : index
              %101 = arith.addi %99, %100 : index
              %102 = arith.addi %101, %94 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %38], LR : [%35, %40]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
              %103 = arith.addi %32, %94 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_6, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%34, %38], LR : [%62, %40]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              %104 = loom.broadcast ins(%27 : memref<64xf16>) outs(%42 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              %105 = loom.broadcast ins(%75 : memref<64xf16>) outs(%77 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %75 : memref<64xf16>
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_7, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%34, %38], LR : [%62, %40]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              %106 = loom.broadcast ins(%74 : memref<64xf16>) outs(%79 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %74 : memref<64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %104, %105, %106 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%72 : memref<64x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
                %114 = arith.subf %in_9, %in_10 : f16
                %115 = math.exp %114 : f16
                %116 = arith.mulf %in, %115 : f16
                %117 = arith.mulf %116, %in_11 : f16
                linalg.yield %117 : f16
              }
              loom.semaphore_give %79 : memref<32x64xf16>
              loom.semaphore_give %77 : memref<32x64xf16>
              loom.semaphore_give %42 : memref<64x32xf16>
              %107 = arith.addi %94, %31 : index
              %108 = arith.muli %arg11, %c8388608 overflow<nsw> : index
              %109 = arith.muli %107, %c4096 overflow<nsw> : index
              %110 = arith.addi %108, %109 : index
              %111 = arith.muli %21, %c512 : index
              %112 = arith.addi %110, %111 : index
              %113 = arith.addi %112, %53 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%34, %64], LR : [%62, %64]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
              linalg.matmul ins(%72, %81 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf16>)
              loom.semaphore_give %81 : memref<64x32xf16>
              loom.semaphore_give %72 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %27 : memref<64xf16>
            %82 = loom.alloc [1] on @L1 : memref<f16>
            %83 = loom.semaphore_take %82 : memref<f16> -> memref<f16>
            %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
            %84 = arith.addi %37, %c3 : index
            loom.copy %reinterpret_cast_2, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %37], LR : [%c7, %84]) : memref<f16, strided<[], offset: ?>> to memref<f16>
            %85 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %86 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
            %87 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %88 = arith.muli %45, %c4096 overflow<nsw> : index
            %89 = arith.addi %87, %88 : index
            %90 = arith.muli %21, %c512 : index
            %91 = arith.addi %89, %90 : index
            %92 = arith.addi %91, %53 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%35, %64], LR : [%35, %64]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %86, %83 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%86 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %94 = arith.mulf %in_5, %in_6 : f16
              %95 = arith.addf %in, %94 : f16
              linalg.yield %95 : f16
            }
            loom.semaphore_give %83 : memref<f16>
            loom.semaphore_give %67 : memref<64x32xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            %93 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
            loom.sync ins(%86 : memref<64x32xf16>) outs(%93 : memref<64x32xf16>)
            loom.copy %93, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%35, %64], LR : [%35, %64]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %93 : memref<64x32xf16>
            loom.semaphore_give %86 : memref<64x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c4, %c2, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg8, %20 : index
          %22 = arith.muli %21, %c8 : index
          %23 = arith.muli %arg9, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %27 = arith.muli %21, %c16384 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg11, %c4 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg12, %c2 : index
          %35 = arith.muli %arg8, %c4 : index
          %36 = arith.addi %34, %35 : index
          %37 = arith.addi %34, %c1 : index
          %38 = arith.addi %37, %35 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          %39 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %40 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
          %41 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
          %42 = loom.broadcast ins(%25 : memref<64xf16>) outs(%41 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
          %43 = arith.addi %23, %29 : index
          %44 = arith.divui %22, %c64 : index
          %45 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %46 = loom.semaphore_take %45 : memref<64x64xf16> -> memref<64x64xf16>
          %47 = arith.muli %43, %c64 overflow<nsw> : index
          %48 = arith.addi %26, %47 : index
          %49 = arith.muli %44, %c64 overflow<nsw> : index
          %50 = arith.addi %48, %49 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%50], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          %51 = arith.muli %arg10, %c32 : index
          %52 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %53 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
          %54 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %55 = arith.muli %arg12, %c1048576 : index
          %56 = arith.addi %54, %55 : index
          %57 = arith.muli %21, %c32768 : index
          %58 = arith.addi %56, %57 : index
          %59 = arith.addi %58, %51 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
          %60 = arith.addi %32, %c3 : index
          %61 = arith.addi %arg10, %34 : index
          %62 = arith.addi %61, %35 : index
          loom.copy %reinterpret_cast_1, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%32, %62], LR : [%60, %62]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
          %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          %65 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          loom.matmul ins(%46, %53 : memref<64x64xf16>, memref<64x32xf16>) outs(%64 : memref<64x32xf16>)
          loom.semaphore_give %53 : memref<64x32xf16>
          loom.semaphore_give %46 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %42 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%65 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %92 = math.exp %in_5 : f16
            %93 = arith.mulf %in, %92 : f16
            linalg.yield %93 : f16
          }
          loom.semaphore_give %64 : memref<64x32xf16>
          loom.semaphore_give %41 : memref<64x32xf16>
          %66 = arith.addi %arg9, %c1 : index
          %67 = arith.muli %66, %c64 : index
          %68 = arith.ceildivui %67, %c64 : index
          %69 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %70 = loom.semaphore_take %69 : memref<64x64xf16> -> memref<64x64xf16>
          %71 = loom.alloc [64] on @L1 : memref<64xf16>
          %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
          %73 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
          %74 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %75 = loom.semaphore_take %74 : memref<32x64xf16> -> memref<32x64xf16>
          %76 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %77 = loom.semaphore_take %76 : memref<32x64xf16> -> memref<32x64xf16>
          %78 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %79 = loom.semaphore_take %78 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %68 step %c1 {
            %92 = arith.muli %arg14, %c64 : index
            %93 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %94 = arith.muli %arg12, %c262144 : index
            %95 = arith.addi %93, %94 : index
            %96 = arith.muli %44, %c65536 overflow<nsw> : index
            %97 = arith.addi %95, %96 : index
            %98 = arith.muli %arg9, %c16384 : index
            %99 = arith.addi %97, %98 : index
            %100 = arith.addi %99, %92 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%100], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %101 = arith.addi %30, %92 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%101], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_6, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%32, %36], LR : [%60, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %102 = loom.broadcast ins(%25 : memref<64xf16>) outs(%40 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
            %103 = loom.broadcast ins(%73 : memref<64xf16>) outs(%75 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %73 : memref<64xf16>
            %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%101], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%32, %36], LR : [%60, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %104 = loom.broadcast ins(%72 : memref<64xf16>) outs(%77 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %72 : memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %102, %103, %104 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%70 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
              %112 = arith.subf %in_9, %in_10 : f16
              %113 = math.exp %112 : f16
              %114 = arith.mulf %in, %113 : f16
              %115 = arith.mulf %114, %in_11 : f16
              linalg.yield %115 : f16
            }
            loom.semaphore_give %77 : memref<32x64xf16>
            loom.semaphore_give %75 : memref<32x64xf16>
            loom.semaphore_give %40 : memref<64x32xf16>
            %105 = arith.addi %92, %29 : index
            %106 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %107 = arith.muli %105, %c4096 overflow<nsw> : index
            %108 = arith.addi %106, %107 : index
            %109 = arith.muli %21, %c512 : index
            %110 = arith.addi %108, %109 : index
            %111 = arith.addi %110, %51 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%111], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_8, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%32, %62], LR : [%60, %62]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%70, %79 : memref<64x64xf16>, memref<64x32xf16>) outs(%65 : memref<64x32xf16>)
            loom.semaphore_give %79 : memref<64x32xf16>
            loom.semaphore_give %70 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %80 = loom.alloc [1] on @L1 : memref<f16>
          %81 = loom.semaphore_take %80 : memref<f16> -> memref<f16>
          %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %82 = arith.addi %35, %c3 : index
          loom.copy %reinterpret_cast_2, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %35], LR : [%c7, %82]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %83 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %84 = loom.semaphore_take %83 : memref<64x32xf16> -> memref<64x32xf16>
          %85 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %86 = arith.muli %43, %c4096 overflow<nsw> : index
          %87 = arith.addi %85, %86 : index
          %88 = arith.muli %21, %c512 : index
          %89 = arith.addi %87, %88 : index
          %90 = arith.addi %89, %51 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%90], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_3, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %62], LR : [%33, %62]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %84, %81 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%84 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %92 = arith.mulf %in_5, %in_6 : f16
            %93 = arith.addf %in, %92 : f16
            linalg.yield %93 : f16
          }
          loom.semaphore_give %81 : memref<f16>
          loom.semaphore_give %65 : memref<64x32xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%90], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %91 = loom.semaphore_take %83 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%84 : memref<64x32xf16>) outs(%91 : memref<64x32xf16>)
          loom.copy %91, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %62], LR : [%33, %62]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %91 : memref<64x32xf16>
          loom.semaphore_give %84 : memref<64x32xf16>
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c4, %c2, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg8, %20 : index
          %22 = arith.muli %21, %c8 : index
          %23 = arith.muli %arg9, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %27 = arith.muli %21, %c16384 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg12, %c4 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg11, %c2 : index
          %35 = arith.muli %arg8, %c4 : index
          %36 = arith.addi %34, %35 : index
          %37 = arith.addi %34, %c1 : index
          %38 = arith.addi %37, %35 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          %39 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %40 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
          %41 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
          %42 = loom.broadcast ins(%25 : memref<64xf16>) outs(%41 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
          %43 = arith.addi %23, %29 : index
          %44 = arith.divui %22, %c64 : index
          %45 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %46 = loom.semaphore_take %45 : memref<64x64xf16> -> memref<64x64xf16>
          %47 = arith.muli %43, %c64 overflow<nsw> : index
          %48 = arith.addi %26, %47 : index
          %49 = arith.muli %44, %c64 overflow<nsw> : index
          %50 = arith.addi %48, %49 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%50], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          %51 = arith.muli %arg10, %c32 : index
          %52 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %53 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
          %54 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %55 = arith.muli %arg12, %c1048576 : index
          %56 = arith.addi %54, %55 : index
          %57 = arith.muli %21, %c32768 : index
          %58 = arith.addi %56, %57 : index
          %59 = arith.addi %58, %51 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
          %60 = arith.addi %32, %c3 : index
          %61 = arith.addi %arg10, %34 : index
          %62 = arith.addi %61, %35 : index
          loom.copy %reinterpret_cast_1, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%32, %62], LR : [%60, %62]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
          %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          %65 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
          loom.matmul ins(%46, %53 : memref<64x64xf16>, memref<64x32xf16>) outs(%64 : memref<64x32xf16>)
          loom.semaphore_give %53 : memref<64x32xf16>
          loom.semaphore_give %46 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %42 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%65 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %92 = math.exp %in_5 : f16
            %93 = arith.mulf %in, %92 : f16
            linalg.yield %93 : f16
          }
          loom.semaphore_give %64 : memref<64x32xf16>
          loom.semaphore_give %41 : memref<64x32xf16>
          %66 = arith.addi %arg9, %c1 : index
          %67 = arith.muli %66, %c64 : index
          %68 = arith.ceildivui %67, %c64 : index
          %69 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %70 = loom.semaphore_take %69 : memref<64x64xf16> -> memref<64x64xf16>
          %71 = loom.alloc [64] on @L1 : memref<64xf16>
          %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
          %73 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
          %74 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %75 = loom.semaphore_take %74 : memref<32x64xf16> -> memref<32x64xf16>
          %76 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %77 = loom.semaphore_take %76 : memref<32x64xf16> -> memref<32x64xf16>
          %78 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %79 = loom.semaphore_take %78 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %68 step %c1 {
            %92 = arith.muli %arg14, %c64 : index
            %93 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %94 = arith.muli %arg12, %c262144 : index
            %95 = arith.addi %93, %94 : index
            %96 = arith.muli %44, %c65536 overflow<nsw> : index
            %97 = arith.addi %95, %96 : index
            %98 = arith.muli %arg9, %c16384 : index
            %99 = arith.addi %97, %98 : index
            %100 = arith.addi %99, %92 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%100], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %101 = arith.addi %30, %92 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%101], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_6, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%32, %36], LR : [%60, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %102 = loom.broadcast ins(%25 : memref<64xf16>) outs(%40 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
            %103 = loom.broadcast ins(%73 : memref<64xf16>) outs(%75 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %73 : memref<64xf16>
            %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%101], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%32, %36], LR : [%60, %38]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %104 = loom.broadcast ins(%72 : memref<64xf16>) outs(%77 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %72 : memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %102, %103, %104 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%70 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
              %112 = arith.subf %in_9, %in_10 : f16
              %113 = math.exp %112 : f16
              %114 = arith.mulf %in, %113 : f16
              %115 = arith.mulf %114, %in_11 : f16
              linalg.yield %115 : f16
            }
            loom.semaphore_give %77 : memref<32x64xf16>
            loom.semaphore_give %75 : memref<32x64xf16>
            loom.semaphore_give %40 : memref<64x32xf16>
            %105 = arith.addi %92, %29 : index
            %106 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %107 = arith.muli %105, %c4096 overflow<nsw> : index
            %108 = arith.addi %106, %107 : index
            %109 = arith.muli %21, %c512 : index
            %110 = arith.addi %108, %109 : index
            %111 = arith.addi %110, %51 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%111], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_8, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%32, %62], LR : [%60, %62]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%70, %79 : memref<64x64xf16>, memref<64x32xf16>) outs(%65 : memref<64x32xf16>)
            loom.semaphore_give %79 : memref<64x32xf16>
            loom.semaphore_give %70 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %25 : memref<64xf16>
          %80 = loom.alloc [1] on @L1 : memref<f16>
          %81 = loom.semaphore_take %80 : memref<f16> -> memref<f16>
          %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %82 = arith.addi %35, %c3 : index
          loom.copy %reinterpret_cast_2, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %35], LR : [%c7, %82]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %83 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %84 = loom.semaphore_take %83 : memref<64x32xf16> -> memref<64x32xf16>
          %85 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %86 = arith.muli %43, %c4096 overflow<nsw> : index
          %87 = arith.addi %85, %86 : index
          %88 = arith.muli %21, %c512 : index
          %89 = arith.addi %87, %88 : index
          %90 = arith.addi %89, %51 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%90], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_3, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %62], LR : [%33, %62]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %84, %81 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%84 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %92 = arith.mulf %in_5, %in_6 : f16
            %93 = arith.addf %in, %92 : f16
            linalg.yield %93 : f16
          }
          loom.semaphore_give %81 : memref<f16>
          loom.semaphore_give %65 : memref<64x32xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%90], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          %91 = loom.semaphore_take %83 : memref<64x32xf16> -> memref<64x32xf16>
          loom.sync ins(%84 : memref<64x32xf16>) outs(%91 : memref<64x32xf16>)
          loom.copy %91, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %62], LR : [%33, %62]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %91 : memref<64x32xf16>
          loom.semaphore_give %84 : memref<64x32xf16>
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c2, %c4) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          scf.for %arg14 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg13, %c2 overflow<nsw> : index
            %21 = arith.addi %arg8, %20 : index
            %22 = arith.muli %arg14, %c2 overflow<nsw> : index
            %23 = arith.addi %arg9, %22 : index
            %24 = arith.muli %21, %c8 : index
            %25 = arith.muli %23, %c64 : index
            %26 = loom.alloc [64] on @L1 : memref<64xf16>
            %27 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %28 = arith.muli %arg11, %c131072 overflow<nsw> : index
            %29 = arith.muli %21, %c16384 : index
            %30 = arith.addi %28, %29 : index
            %31 = arith.muli %arg12, %c1024 : index
            %32 = arith.addi %30, %31 : index
            %33 = arith.addi %32, %25 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            %34 = arith.muli %arg11, %c2 : index
            %35 = arith.addi %arg9, %34 : index
            %36 = arith.muli %arg8, %c4 : index
            %37 = arith.addi %35, %36 : index
            %38 = arith.muli %arg12, %c2 : index
            %39 = arith.addi %38, %c1 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%37, %38], LR : [%37, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %40 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %41 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
            %42 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.broadcast ins(%27 : memref<64xf16>) outs(%42 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            %44 = arith.addi %25, %31 : index
            %45 = arith.divui %24, %c64 : index
            %46 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %47 = loom.semaphore_take %46 : memref<64x64xf16> -> memref<64x64xf16>
            %48 = arith.muli %44, %c64 overflow<nsw> : index
            %49 = arith.addi %28, %48 : index
            %50 = arith.muli %45, %c64 overflow<nsw> : index
            %51 = arith.addi %49, %50 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%37, %38], LR : [%37, %39]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
            %52 = arith.muli %arg10, %c32 : index
            %53 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %54 = loom.semaphore_take %53 : memref<64x32xf16> -> memref<64x32xf16>
            %55 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %56 = arith.muli %arg12, %c1048576 : index
            %57 = arith.addi %55, %56 : index
            %58 = arith.muli %21, %c32768 : index
            %59 = arith.addi %57, %58 : index
            %60 = arith.addi %59, %52 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
            %61 = arith.addi %34, %36 : index
            %62 = arith.addi %34, %c1 : index
            %63 = arith.addi %62, %36 : index
            %64 = arith.addi %arg10, %38 : index
            loom.copy %reinterpret_cast_1, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%61, %64], LR : [%63, %64]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
            %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %66 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
            %67 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%47, %54 : memref<64x64xf16>, memref<64x32xf16>) outs(%66 : memref<64x32xf16>)
            loom.semaphore_give %54 : memref<64x32xf16>
            loom.semaphore_give %47 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %43 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%67 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %94 = math.exp %in_5 : f16
              %95 = arith.mulf %in, %94 : f16
              linalg.yield %95 : f16
            }
            loom.semaphore_give %66 : memref<64x32xf16>
            loom.semaphore_give %42 : memref<64x32xf16>
            %68 = arith.addi %23, %c1 : index
            %69 = arith.muli %68, %c64 : index
            %70 = arith.ceildivui %69, %c64 : index
            %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
            %73 = loom.alloc [64] on @L1 : memref<64xf16>
            %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
            %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
            %76 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %77 = loom.semaphore_take %76 : memref<32x64xf16> -> memref<32x64xf16>
            %78 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %79 = loom.semaphore_take %78 : memref<32x64xf16> -> memref<32x64xf16>
            %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
            scf.for %arg15 = %c0 to %70 step %c1 {
              %94 = arith.muli %arg15, %c64 : index
              %95 = arith.muli %arg11, %c524288 overflow<nsw> : index
              %96 = arith.muli %arg12, %c262144 : index
              %97 = arith.addi %95, %96 : index
              %98 = arith.muli %45, %c65536 overflow<nsw> : index
              %99 = arith.addi %97, %98 : index
              %100 = arith.muli %23, %c16384 : index
              %101 = arith.addi %99, %100 : index
              %102 = arith.addi %101, %94 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%37, %38], LR : [%37, %39]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
              %103 = arith.addi %32, %94 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_6, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%61, %38], LR : [%63, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              %104 = loom.broadcast ins(%27 : memref<64xf16>) outs(%41 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              %105 = loom.broadcast ins(%75 : memref<64xf16>) outs(%77 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %75 : memref<64xf16>
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_7, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%61, %38], LR : [%63, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              %106 = loom.broadcast ins(%74 : memref<64xf16>) outs(%79 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %74 : memref<64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %104, %105, %106 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%72 : memref<64x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
                %114 = arith.subf %in_9, %in_10 : f16
                %115 = math.exp %114 : f16
                %116 = arith.mulf %in, %115 : f16
                %117 = arith.mulf %116, %in_11 : f16
                linalg.yield %117 : f16
              }
              loom.semaphore_give %79 : memref<32x64xf16>
              loom.semaphore_give %77 : memref<32x64xf16>
              loom.semaphore_give %41 : memref<64x32xf16>
              %107 = arith.addi %94, %31 : index
              %108 = arith.muli %arg11, %c8388608 overflow<nsw> : index
              %109 = arith.muli %107, %c4096 overflow<nsw> : index
              %110 = arith.addi %108, %109 : index
              %111 = arith.muli %21, %c512 : index
              %112 = arith.addi %110, %111 : index
              %113 = arith.addi %112, %52 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%61, %64], LR : [%63, %64]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
              linalg.matmul ins(%72, %81 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf16>)
              loom.semaphore_give %81 : memref<64x32xf16>
              loom.semaphore_give %72 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %27 : memref<64xf16>
            %82 = loom.alloc [1] on @L1 : memref<f16>
            %83 = loom.semaphore_take %82 : memref<f16> -> memref<f16>
            %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
            %84 = arith.addi %36, %c3 : index
            loom.copy %reinterpret_cast_2, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%36, %c0], LR : [%84, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
            %85 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %86 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
            %87 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %88 = arith.muli %44, %c4096 overflow<nsw> : index
            %89 = arith.addi %87, %88 : index
            %90 = arith.muli %21, %c512 : index
            %91 = arith.addi %89, %90 : index
            %92 = arith.addi %91, %52 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%37, %64], LR : [%37, %64]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %86, %83 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%86 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %94 = arith.mulf %in_5, %in_6 : f16
              %95 = arith.addf %in, %94 : f16
              linalg.yield %95 : f16
            }
            loom.semaphore_give %83 : memref<f16>
            loom.semaphore_give %67 : memref<64x32xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            %93 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
            loom.sync ins(%86 : memref<64x32xf16>) outs(%93 : memref<64x32xf16>)
            loom.copy %93, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%37, %64], LR : [%37, %64]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %93 : memref<64x32xf16>
            loom.semaphore_give %86 : memref<64x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c4, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          scf.for %arg14 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg13, %c2 overflow<nsw> : index
            %21 = arith.addi %arg8, %20 : index
            %22 = arith.muli %arg14, %c2 overflow<nsw> : index
            %23 = arith.addi %arg9, %22 : index
            %24 = arith.muli %21, %c8 : index
            %25 = arith.muli %23, %c64 : index
            %26 = loom.alloc [64] on @L1 : memref<64xf16>
            %27 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %28 = arith.muli %arg11, %c131072 overflow<nsw> : index
            %29 = arith.muli %21, %c16384 : index
            %30 = arith.addi %28, %29 : index
            %31 = arith.muli %arg12, %c1024 : index
            %32 = arith.addi %30, %31 : index
            %33 = arith.addi %32, %25 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            %34 = arith.muli %arg12, %c2 : index
            %35 = arith.addi %arg9, %34 : index
            %36 = arith.muli %arg8, %c4 : index
            %37 = arith.addi %35, %36 : index
            %38 = arith.muli %arg11, %c2 : index
            %39 = arith.addi %38, %c1 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%37, %38], LR : [%37, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %40 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %41 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
            %42 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.broadcast ins(%27 : memref<64xf16>) outs(%42 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            %44 = arith.addi %25, %31 : index
            %45 = arith.divui %24, %c64 : index
            %46 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %47 = loom.semaphore_take %46 : memref<64x64xf16> -> memref<64x64xf16>
            %48 = arith.muli %44, %c64 overflow<nsw> : index
            %49 = arith.addi %28, %48 : index
            %50 = arith.muli %45, %c64 overflow<nsw> : index
            %51 = arith.addi %49, %50 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%37, %38], LR : [%37, %39]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
            %52 = arith.muli %arg10, %c32 : index
            %53 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %54 = loom.semaphore_take %53 : memref<64x32xf16> -> memref<64x32xf16>
            %55 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %56 = arith.muli %arg12, %c1048576 : index
            %57 = arith.addi %55, %56 : index
            %58 = arith.muli %21, %c32768 : index
            %59 = arith.addi %57, %58 : index
            %60 = arith.addi %59, %52 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
            %61 = arith.addi %34, %36 : index
            %62 = arith.addi %34, %c1 : index
            %63 = arith.addi %62, %36 : index
            %64 = arith.addi %arg10, %38 : index
            loom.copy %reinterpret_cast_1, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%61, %64], LR : [%63, %64]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
            %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %66 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
            %67 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%47, %54 : memref<64x64xf16>, memref<64x32xf16>) outs(%66 : memref<64x32xf16>)
            loom.semaphore_give %54 : memref<64x32xf16>
            loom.semaphore_give %47 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %43 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%67 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %94 = math.exp %in_5 : f16
              %95 = arith.mulf %in, %94 : f16
              linalg.yield %95 : f16
            }
            loom.semaphore_give %66 : memref<64x32xf16>
            loom.semaphore_give %42 : memref<64x32xf16>
            %68 = arith.addi %23, %c1 : index
            %69 = arith.muli %68, %c64 : index
            %70 = arith.ceildivui %69, %c64 : index
            %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
            %73 = loom.alloc [64] on @L1 : memref<64xf16>
            %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
            %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
            %76 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %77 = loom.semaphore_take %76 : memref<32x64xf16> -> memref<32x64xf16>
            %78 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %79 = loom.semaphore_take %78 : memref<32x64xf16> -> memref<32x64xf16>
            %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
            scf.for %arg15 = %c0 to %70 step %c1 {
              %94 = arith.muli %arg15, %c64 : index
              %95 = arith.muli %arg11, %c524288 overflow<nsw> : index
              %96 = arith.muli %arg12, %c262144 : index
              %97 = arith.addi %95, %96 : index
              %98 = arith.muli %45, %c65536 overflow<nsw> : index
              %99 = arith.addi %97, %98 : index
              %100 = arith.muli %23, %c16384 : index
              %101 = arith.addi %99, %100 : index
              %102 = arith.addi %101, %94 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%37, %38], LR : [%37, %39]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
              %103 = arith.addi %32, %94 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_6, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%61, %38], LR : [%63, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              %104 = loom.broadcast ins(%27 : memref<64xf16>) outs(%41 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              %105 = loom.broadcast ins(%75 : memref<64xf16>) outs(%77 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %75 : memref<64xf16>
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_7, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%61, %38], LR : [%63, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              %106 = loom.broadcast ins(%74 : memref<64xf16>) outs(%79 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %74 : memref<64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %104, %105, %106 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%72 : memref<64x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
                %114 = arith.subf %in_9, %in_10 : f16
                %115 = math.exp %114 : f16
                %116 = arith.mulf %in, %115 : f16
                %117 = arith.mulf %116, %in_11 : f16
                linalg.yield %117 : f16
              }
              loom.semaphore_give %79 : memref<32x64xf16>
              loom.semaphore_give %77 : memref<32x64xf16>
              loom.semaphore_give %41 : memref<64x32xf16>
              %107 = arith.addi %94, %31 : index
              %108 = arith.muli %arg11, %c8388608 overflow<nsw> : index
              %109 = arith.muli %107, %c4096 overflow<nsw> : index
              %110 = arith.addi %108, %109 : index
              %111 = arith.muli %21, %c512 : index
              %112 = arith.addi %110, %111 : index
              %113 = arith.addi %112, %52 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%61, %64], LR : [%63, %64]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
              linalg.matmul ins(%72, %81 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf16>)
              loom.semaphore_give %81 : memref<64x32xf16>
              loom.semaphore_give %72 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %27 : memref<64xf16>
            %82 = loom.alloc [1] on @L1 : memref<f16>
            %83 = loom.semaphore_take %82 : memref<f16> -> memref<f16>
            %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
            %84 = arith.addi %36, %c3 : index
            loom.copy %reinterpret_cast_2, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%36, %c0], LR : [%84, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
            %85 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %86 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
            %87 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %88 = arith.muli %44, %c4096 overflow<nsw> : index
            %89 = arith.addi %87, %88 : index
            %90 = arith.muli %21, %c512 : index
            %91 = arith.addi %89, %90 : index
            %92 = arith.addi %91, %52 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%37, %64], LR : [%37, %64]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %86, %83 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%86 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %94 = arith.mulf %in_5, %in_6 : f16
              %95 = arith.addf %in, %94 : f16
              linalg.yield %95 : f16
            }
            loom.semaphore_give %83 : memref<f16>
            loom.semaphore_give %67 : memref<64x32xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            %93 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
            loom.sync ins(%86 : memref<64x32xf16>) outs(%93 : memref<64x32xf16>)
            loom.copy %93, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%37, %64], LR : [%37, %64]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %93 : memref<64x32xf16>
            loom.semaphore_give %86 : memref<64x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc4_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c4, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          scf.for %arg14 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg13, %c2 overflow<nsw> : index
            %21 = arith.addi %arg8, %20 : index
            %22 = arith.muli %arg14, %c2 overflow<nsw> : index
            %23 = arith.addi %arg9, %22 : index
            %24 = arith.muli %21, %c8 : index
            %25 = arith.muli %23, %c64 : index
            %26 = loom.alloc [64] on @L1 : memref<64xf16>
            %27 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %28 = arith.muli %arg11, %c131072 overflow<nsw> : index
            %29 = arith.muli %21, %c16384 : index
            %30 = arith.addi %28, %29 : index
            %31 = arith.muli %arg12, %c1024 : index
            %32 = arith.addi %30, %31 : index
            %33 = arith.addi %32, %25 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            %34 = arith.muli %arg11, %c2 : index
            %35 = arith.addi %arg9, %34 : index
            %36 = arith.muli %arg8, %c4 : index
            %37 = arith.addi %35, %36 : index
            %38 = arith.muli %arg12, %c4 : index
            %39 = arith.addi %38, %c3 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%37, %38], LR : [%37, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %40 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %41 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
            %42 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.broadcast ins(%27 : memref<64xf16>) outs(%42 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            %44 = arith.addi %25, %31 : index
            %45 = arith.divui %24, %c64 : index
            %46 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %47 = loom.semaphore_take %46 : memref<64x64xf16> -> memref<64x64xf16>
            %48 = arith.muli %44, %c64 overflow<nsw> : index
            %49 = arith.addi %28, %48 : index
            %50 = arith.muli %45, %c64 overflow<nsw> : index
            %51 = arith.addi %49, %50 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%37, %38], LR : [%37, %39]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
            %52 = arith.muli %arg10, %c32 : index
            %53 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %54 = loom.semaphore_take %53 : memref<64x32xf16> -> memref<64x32xf16>
            %55 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %56 = arith.muli %arg12, %c1048576 : index
            %57 = arith.addi %55, %56 : index
            %58 = arith.muli %21, %c32768 : index
            %59 = arith.addi %57, %58 : index
            %60 = arith.addi %59, %52 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
            %61 = arith.addi %34, %36 : index
            %62 = arith.addi %34, %c1 : index
            %63 = arith.addi %62, %36 : index
            %64 = arith.addi %arg10, %38 : index
            loom.copy %reinterpret_cast_1, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%61, %64], LR : [%63, %64]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
            %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %66 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
            %67 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%47, %54 : memref<64x64xf16>, memref<64x32xf16>) outs(%66 : memref<64x32xf16>)
            loom.semaphore_give %54 : memref<64x32xf16>
            loom.semaphore_give %47 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %43 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%67 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %94 = math.exp %in_5 : f16
              %95 = arith.mulf %in, %94 : f16
              linalg.yield %95 : f16
            }
            loom.semaphore_give %66 : memref<64x32xf16>
            loom.semaphore_give %42 : memref<64x32xf16>
            %68 = arith.addi %23, %c1 : index
            %69 = arith.muli %68, %c64 : index
            %70 = arith.ceildivui %69, %c64 : index
            %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
            %73 = loom.alloc [64] on @L1 : memref<64xf16>
            %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
            %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
            %76 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %77 = loom.semaphore_take %76 : memref<32x64xf16> -> memref<32x64xf16>
            %78 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %79 = loom.semaphore_take %78 : memref<32x64xf16> -> memref<32x64xf16>
            %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
            scf.for %arg15 = %c0 to %70 step %c1 {
              %94 = arith.muli %arg15, %c64 : index
              %95 = arith.muli %arg11, %c524288 overflow<nsw> : index
              %96 = arith.muli %arg12, %c262144 : index
              %97 = arith.addi %95, %96 : index
              %98 = arith.muli %45, %c65536 overflow<nsw> : index
              %99 = arith.addi %97, %98 : index
              %100 = arith.muli %23, %c16384 : index
              %101 = arith.addi %99, %100 : index
              %102 = arith.addi %101, %94 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%37, %38], LR : [%37, %39]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
              %103 = arith.addi %32, %94 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_6, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%61, %38], LR : [%63, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              %104 = loom.broadcast ins(%27 : memref<64xf16>) outs(%41 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              %105 = loom.broadcast ins(%75 : memref<64xf16>) outs(%77 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %75 : memref<64xf16>
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_7, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%61, %38], LR : [%63, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              %106 = loom.broadcast ins(%74 : memref<64xf16>) outs(%79 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %74 : memref<64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %104, %105, %106 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%72 : memref<64x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
                %114 = arith.subf %in_9, %in_10 : f16
                %115 = math.exp %114 : f16
                %116 = arith.mulf %in, %115 : f16
                %117 = arith.mulf %116, %in_11 : f16
                linalg.yield %117 : f16
              }
              loom.semaphore_give %79 : memref<32x64xf16>
              loom.semaphore_give %77 : memref<32x64xf16>
              loom.semaphore_give %41 : memref<64x32xf16>
              %107 = arith.addi %94, %31 : index
              %108 = arith.muli %arg11, %c8388608 overflow<nsw> : index
              %109 = arith.muli %107, %c4096 overflow<nsw> : index
              %110 = arith.addi %108, %109 : index
              %111 = arith.muli %21, %c512 : index
              %112 = arith.addi %110, %111 : index
              %113 = arith.addi %112, %52 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%61, %64], LR : [%63, %64]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
              linalg.matmul ins(%72, %81 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf16>)
              loom.semaphore_give %81 : memref<64x32xf16>
              loom.semaphore_give %72 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %27 : memref<64xf16>
            %82 = loom.alloc [1] on @L1 : memref<f16>
            %83 = loom.semaphore_take %82 : memref<f16> -> memref<f16>
            %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
            %84 = arith.addi %36, %c3 : index
            loom.copy %reinterpret_cast_2, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%36, %c0], LR : [%84, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
            %85 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %86 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
            %87 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %88 = arith.muli %44, %c4096 overflow<nsw> : index
            %89 = arith.addi %87, %88 : index
            %90 = arith.muli %21, %c512 : index
            %91 = arith.addi %89, %90 : index
            %92 = arith.addi %91, %52 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%37, %64], LR : [%37, %64]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %86, %83 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%86 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %94 = arith.mulf %in_5, %in_6 : f16
              %95 = arith.addf %in, %94 : f16
              linalg.yield %95 : f16
            }
            loom.semaphore_give %83 : memref<f16>
            loom.semaphore_give %67 : memref<64x32xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            %93 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
            loom.sync ins(%86 : memref<64x32xf16>) outs(%93 : memref<64x32xf16>)
            loom.copy %93, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%37, %64], LR : [%37, %64]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %93 : memref<64x32xf16>
            loom.semaphore_give %86 : memref<64x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c4, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          scf.for %arg14 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg13, %c2 overflow<nsw> : index
            %21 = arith.addi %arg8, %20 : index
            %22 = arith.muli %arg14, %c2 overflow<nsw> : index
            %23 = arith.addi %arg9, %22 : index
            %24 = arith.muli %21, %c8 : index
            %25 = arith.muli %23, %c64 : index
            %26 = loom.alloc [64] on @L1 : memref<64xf16>
            %27 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %28 = arith.muli %arg11, %c131072 overflow<nsw> : index
            %29 = arith.muli %21, %c16384 : index
            %30 = arith.addi %28, %29 : index
            %31 = arith.muli %arg12, %c1024 : index
            %32 = arith.addi %30, %31 : index
            %33 = arith.addi %32, %25 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            %34 = arith.muli %arg12, %c2 : index
            %35 = arith.addi %arg9, %34 : index
            %36 = arith.muli %arg8, %c4 : index
            %37 = arith.addi %35, %36 : index
            %38 = arith.muli %arg11, %c4 : index
            %39 = arith.addi %38, %c3 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%37, %38], LR : [%37, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            %40 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %41 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
            %42 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.broadcast ins(%27 : memref<64xf16>) outs(%42 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            %44 = arith.addi %25, %31 : index
            %45 = arith.divui %24, %c64 : index
            %46 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %47 = loom.semaphore_take %46 : memref<64x64xf16> -> memref<64x64xf16>
            %48 = arith.muli %44, %c64 overflow<nsw> : index
            %49 = arith.addi %28, %48 : index
            %50 = arith.muli %45, %c64 overflow<nsw> : index
            %51 = arith.addi %49, %50 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%37, %38], LR : [%37, %39]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
            %52 = arith.muli %arg10, %c32 : index
            %53 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %54 = loom.semaphore_take %53 : memref<64x32xf16> -> memref<64x32xf16>
            %55 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %56 = arith.muli %arg12, %c1048576 : index
            %57 = arith.addi %55, %56 : index
            %58 = arith.muli %21, %c32768 : index
            %59 = arith.addi %57, %58 : index
            %60 = arith.addi %59, %52 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
            %61 = arith.addi %34, %36 : index
            %62 = arith.addi %34, %c1 : index
            %63 = arith.addi %62, %36 : index
            %64 = arith.addi %arg10, %38 : index
            loom.copy %reinterpret_cast_1, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%61, %64], LR : [%63, %64]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
            %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %66 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
            %67 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%47, %54 : memref<64x64xf16>, memref<64x32xf16>) outs(%66 : memref<64x32xf16>)
            loom.semaphore_give %54 : memref<64x32xf16>
            loom.semaphore_give %47 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %43 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%67 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %94 = math.exp %in_5 : f16
              %95 = arith.mulf %in, %94 : f16
              linalg.yield %95 : f16
            }
            loom.semaphore_give %66 : memref<64x32xf16>
            loom.semaphore_give %42 : memref<64x32xf16>
            %68 = arith.addi %23, %c1 : index
            %69 = arith.muli %68, %c64 : index
            %70 = arith.ceildivui %69, %c64 : index
            %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
            %73 = loom.alloc [64] on @L1 : memref<64xf16>
            %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
            %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
            %76 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %77 = loom.semaphore_take %76 : memref<32x64xf16> -> memref<32x64xf16>
            %78 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %79 = loom.semaphore_take %78 : memref<32x64xf16> -> memref<32x64xf16>
            %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
            scf.for %arg15 = %c0 to %70 step %c1 {
              %94 = arith.muli %arg15, %c64 : index
              %95 = arith.muli %arg11, %c524288 overflow<nsw> : index
              %96 = arith.muli %arg12, %c262144 : index
              %97 = arith.addi %95, %96 : index
              %98 = arith.muli %45, %c65536 overflow<nsw> : index
              %99 = arith.addi %97, %98 : index
              %100 = arith.muli %23, %c16384 : index
              %101 = arith.addi %99, %100 : index
              %102 = arith.addi %101, %94 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%37, %38], LR : [%37, %39]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
              %103 = arith.addi %32, %94 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_6, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%61, %38], LR : [%63, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              %104 = loom.broadcast ins(%27 : memref<64xf16>) outs(%41 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              %105 = loom.broadcast ins(%75 : memref<64xf16>) outs(%77 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %75 : memref<64xf16>
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_7, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%61, %38], LR : [%63, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              %106 = loom.broadcast ins(%74 : memref<64xf16>) outs(%79 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %74 : memref<64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %104, %105, %106 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%72 : memref<64x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
                %114 = arith.subf %in_9, %in_10 : f16
                %115 = math.exp %114 : f16
                %116 = arith.mulf %in, %115 : f16
                %117 = arith.mulf %116, %in_11 : f16
                linalg.yield %117 : f16
              }
              loom.semaphore_give %79 : memref<32x64xf16>
              loom.semaphore_give %77 : memref<32x64xf16>
              loom.semaphore_give %41 : memref<64x32xf16>
              %107 = arith.addi %94, %31 : index
              %108 = arith.muli %arg11, %c8388608 overflow<nsw> : index
              %109 = arith.muli %107, %c4096 overflow<nsw> : index
              %110 = arith.addi %108, %109 : index
              %111 = arith.muli %21, %c512 : index
              %112 = arith.addi %110, %111 : index
              %113 = arith.addi %112, %52 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%61, %64], LR : [%63, %64]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
              linalg.matmul ins(%72, %81 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf16>)
              loom.semaphore_give %81 : memref<64x32xf16>
              loom.semaphore_give %72 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %27 : memref<64xf16>
            %82 = loom.alloc [1] on @L1 : memref<f16>
            %83 = loom.semaphore_take %82 : memref<f16> -> memref<f16>
            %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
            %84 = arith.addi %36, %c3 : index
            loom.copy %reinterpret_cast_2, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%36, %c0], LR : [%84, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
            %85 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %86 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
            %87 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %88 = arith.muli %44, %c4096 overflow<nsw> : index
            %89 = arith.addi %87, %88 : index
            %90 = arith.muli %21, %c512 : index
            %91 = arith.addi %89, %90 : index
            %92 = arith.addi %91, %52 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%37, %64], LR : [%37, %64]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %86, %83 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%86 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %94 = arith.mulf %in_5, %in_6 : f16
              %95 = arith.addf %in, %94 : f16
              linalg.yield %95 : f16
            }
            loom.semaphore_give %83 : memref<f16>
            loom.semaphore_give %67 : memref<64x32xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            %93 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
            loom.sync ins(%86 : memref<64x32xf16>) outs(%93 : memref<64x32xf16>)
            loom.copy %93, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%37, %64], LR : [%37, %64]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %93 : memref<64x32xf16>
            loom.semaphore_give %86 : memref<64x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
}
