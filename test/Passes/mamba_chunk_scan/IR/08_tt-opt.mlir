module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c512 = arith.constant 512 : index
      %c8192 = arith.constant 8192 : index
      %c1024 = arith.constant 1024 : index
      %c16384 = arith.constant 16384 : index
      %c2097152 = arith.constant 2097152 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c131072 = arith.constant 131072 : index
      %c32768 = arith.constant 32768 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c4, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg9, %20 : index
          %22 = arith.muli %arg8, %c8 : index
          %23 = arith.muli %21, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c32768 overflow<nsw> : index
          %27 = arith.muli %arg8, %c16384 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg11, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg12, %c2 : index
          %35 = arith.muli %arg8, %c4 : index
          %36 = arith.addi %34, %35 : index
          %37 = arith.addi %34, %c1 : index
          %38 = arith.addi %37, %35 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
          %39 = loom.alloc [64] on @L1 : memref<64xf32>
          %40 = loom.semaphore_take %39 : memref<64xf32> -> memref<64xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25 : memref<64xf16>) outs(%40 : memref<64xf32>) {
          ^bb0(%in: f16, %out: f32):
            %93 = arith.extf %in : f16 to f32
            linalg.yield %93 : f32
          }
          loom.semaphore_give %25 : memref<64xf16>
          %41 = loom.alloc [64] on @L1 : memref<64xf32>
          %42 = loom.semaphore_take %41 : memref<64xf32> -> memref<64xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%40 : memref<64xf32>) outs(%42 : memref<64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %93 = arith.truncf %cst_0 : f64 to f32
            %94 = arith.mulf %in, %93 : f32
            %95 = math.powf %cst, %94 : f32
            linalg.yield %95 : f32
          }
          %43 = arith.divui %22, %c16 : index
          %44 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
          %45 = loom.semaphore_take %44 : memref<64x16xf16> -> memref<64x16xf16>
          %46 = arith.muli %arg12, %c16384 : index
          %47 = arith.addi %26, %46 : index
          %48 = arith.muli %43, %c16 overflow<nsw> : index
          %49 = arith.addi %47, %48 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%49], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
          %50 = arith.addi %32, %c1 : index
          loom.copy %reinterpret_cast_1, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%50, %38]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
          %51 = arith.muli %arg10, %c32 : index
          %52 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
          %53 = loom.semaphore_take %52 : memref<32x16xf16> -> memref<32x16xf16>
          %54 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %55 = arith.muli %arg12, %c65536 : index
          %56 = arith.addi %54, %55 : index
          %57 = arith.muli %arg8, %c8192 : index
          %58 = arith.addi %56, %57 : index
          %59 = arith.muli %arg10, %c512 : index
          %60 = arith.addi %58, %59 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
          %61 = arith.addi %arg10, %34 : index
          %62 = arith.addi %61, %35 : index
          loom.copy %reinterpret_cast_2, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %62], LR : [%50, %62]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
          %63 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
          %64 = loom.semaphore_take %63 : memref<16x32xf16> -> memref<16x32xf16>
          linalg.transpose ins(%53 : memref<32x16xf16>) outs(%64 : memref<16x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %53 : memref<32x16xf16>
          %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
          %66 = loom.semaphore_take %65 : memref<64x32xf32> -> memref<64x32xf32>
          %67 = loom.semaphore_take %65 : memref<64x32xf32> -> memref<64x32xf32>
          loom.matmul ins(%45, %64 : memref<64x16xf16>, memref<16x32xf16>) outs(%66 : memref<64x32xf32>)
          loom.semaphore_give %64 : memref<16x32xf16>
          loom.semaphore_give %45 : memref<64x16xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %42 : memref<64x32xf32>, memref<64xf32>) outs(%67 : memref<64x32xf32>) {
          ^bb0(%in: f32, %in_6: f32, %out: f32):
            %93 = arith.mulf %in, %in_6 : f32
            linalg.yield %93 : f32
          }
          loom.semaphore_give %66 : memref<64x32xf32>
          loom.semaphore_give %42 : memref<64xf32>
          %68 = arith.addi %21, %c1 : index
          %69 = arith.muli %68, %c64 : index
          %70 = arith.ceildivui %69, %c64 : index
          %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
          %73 = loom.alloc [64] on @L1 : memref<64xf16>
          %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %76 = loom.alloc [64] on @L1 : memref<64xf32>
          %77 = loom.semaphore_take %76 : memref<64xf32> -> memref<64xf32>
          %78 = loom.alloc [64] on @L1 : memref<64xf32>
          %79 = loom.semaphore_take %78 : memref<64xf32> -> memref<64xf32>
          %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %70 step %c1 {
            %93 = arith.muli %arg14, %c64 : index
            %94 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %95 = arith.muli %arg12, %c262144 : index
            %96 = arith.addi %94, %95 : index
            %97 = arith.muli %43, %c65536 overflow<nsw> : index
            %98 = arith.addi %96, %97 : index
            %99 = arith.muli %21, %c16384 : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.addi %100, %93 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%101], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %102 = arith.addi %30, %93 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%50, %38]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%75 : memref<64xf16>) outs(%77 : memref<64xf32>) {
            ^bb0(%in: f16, %out: f32):
              %109 = arith.extf %in : f16 to f32
              linalg.yield %109 : f32
            }
            loom.semaphore_give %75 : memref<64xf16>
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%50, %38]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%74 : memref<64xf16>) outs(%79 : memref<64xf32>) {
            ^bb0(%in: f16, %out: f32):
              %109 = arith.extf %in : f16 to f32
              linalg.yield %109 : f32
            }
            loom.semaphore_give %74 : memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %40, %77, %79 : memref<64x64xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%72 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_10: f32, %in_11: f32, %in_12: f32, %out: f16):
              %109 = arith.truncf %cst_0 : f64 to f32
              %110 = arith.mulf %in_11, %109 : f32
              %111 = arith.mulf %in_10, %109 : f32
              %112 = arith.subf %111, %110 : f32
              %113 = math.powf %cst, %112 : f32
              %114 = arith.extf %in : f16 to f32
              %115 = arith.mulf %114, %113 : f32
              %116 = arith.mulf %115, %in_12 : f32
              %117 = arith.truncf %116 : f32 to f16
              linalg.yield %117 : f16
            }
            loom.semaphore_give %79 : memref<64xf32>
            loom.semaphore_give %77 : memref<64xf32>
            %103 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %104 = arith.muli %arg12, %c1048576 : index
            %105 = arith.addi %103, %104 : index
            %106 = arith.muli %arg8, %c512 : index
            %107 = arith.addi %105, %106 : index
            %108 = arith.addi %107, %51 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%108], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %62], LR : [%50, %62]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%72, %81 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf32>)
            loom.semaphore_give %81 : memref<64x32xf16>
            loom.semaphore_give %72 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %40 : memref<64xf32>
          %82 = loom.alloc [1] on @L1 : memref<f16>
          %83 = loom.semaphore_take %82 : memref<f16> -> memref<f16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
          %84 = arith.addi %35, %c3 : index
          loom.copy %reinterpret_cast_3, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %35], LR : [%c7, %84]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %85 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %86 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
          %87 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %88 = arith.muli %arg12, %c1048576 : index
          %89 = arith.addi %87, %88 : index
          %90 = arith.muli %arg8, %c512 : index
          %91 = arith.addi %89, %90 : index
          %92 = arith.addi %91, %51 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %62], LR : [%50, %62]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %86, %83 : memref<64x32xf32>, memref<64x32xf16>, memref<f16>) outs(%86 : memref<64x32xf16>) {
          ^bb0(%in: f32, %in_6: f16, %in_7: f16, %out: f16):
            %93 = arith.extf %in_7 : f16 to f32
            %94 = arith.extf %in_6 : f16 to f32
            %95 = arith.mulf %94, %93 : f32
            %96 = arith.addf %in, %95 : f32
            %97 = arith.truncf %96 : f32 to f16
            linalg.yield %97 : f16
          }
          loom.semaphore_give %83 : memref<f16>
          loom.semaphore_give %67 : memref<64x32xf32>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %86, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%32, %62], LR : [%50, %62]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.semaphore_give %86 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c512 = arith.constant 512 : index
      %c8192 = arith.constant 8192 : index
      %c1024 = arith.constant 1024 : index
      %c16384 = arith.constant 16384 : index
      %c2097152 = arith.constant 2097152 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c131072 = arith.constant 131072 : index
      %c32768 = arith.constant 32768 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c2, %c4) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg9, %20 : index
          %22 = arith.muli %arg8, %c8 : index
          %23 = arith.muli %21, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c32768 overflow<nsw> : index
          %27 = arith.muli %arg8, %c16384 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg12, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg11, %c2 : index
          %35 = arith.muli %arg8, %c4 : index
          %36 = arith.addi %34, %35 : index
          %37 = arith.addi %34, %c1 : index
          %38 = arith.addi %37, %35 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
          %39 = loom.alloc [64] on @L1 : memref<64xf32>
          %40 = loom.semaphore_take %39 : memref<64xf32> -> memref<64xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25 : memref<64xf16>) outs(%40 : memref<64xf32>) {
          ^bb0(%in: f16, %out: f32):
            %93 = arith.extf %in : f16 to f32
            linalg.yield %93 : f32
          }
          loom.semaphore_give %25 : memref<64xf16>
          %41 = loom.alloc [64] on @L1 : memref<64xf32>
          %42 = loom.semaphore_take %41 : memref<64xf32> -> memref<64xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%40 : memref<64xf32>) outs(%42 : memref<64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %93 = arith.truncf %cst_0 : f64 to f32
            %94 = arith.mulf %in, %93 : f32
            %95 = math.powf %cst, %94 : f32
            linalg.yield %95 : f32
          }
          %43 = arith.divui %22, %c16 : index
          %44 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
          %45 = loom.semaphore_take %44 : memref<64x16xf16> -> memref<64x16xf16>
          %46 = arith.muli %arg12, %c16384 : index
          %47 = arith.addi %26, %46 : index
          %48 = arith.muli %43, %c16 overflow<nsw> : index
          %49 = arith.addi %47, %48 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%49], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
          %50 = arith.addi %32, %c1 : index
          loom.copy %reinterpret_cast_1, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%50, %38]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
          %51 = arith.muli %arg10, %c32 : index
          %52 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
          %53 = loom.semaphore_take %52 : memref<32x16xf16> -> memref<32x16xf16>
          %54 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %55 = arith.muli %arg12, %c65536 : index
          %56 = arith.addi %54, %55 : index
          %57 = arith.muli %arg8, %c8192 : index
          %58 = arith.addi %56, %57 : index
          %59 = arith.muli %arg10, %c512 : index
          %60 = arith.addi %58, %59 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
          %61 = arith.addi %arg10, %34 : index
          %62 = arith.addi %61, %35 : index
          loom.copy %reinterpret_cast_2, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %62], LR : [%50, %62]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
          %63 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
          %64 = loom.semaphore_take %63 : memref<16x32xf16> -> memref<16x32xf16>
          linalg.transpose ins(%53 : memref<32x16xf16>) outs(%64 : memref<16x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %53 : memref<32x16xf16>
          %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
          %66 = loom.semaphore_take %65 : memref<64x32xf32> -> memref<64x32xf32>
          %67 = loom.semaphore_take %65 : memref<64x32xf32> -> memref<64x32xf32>
          loom.matmul ins(%45, %64 : memref<64x16xf16>, memref<16x32xf16>) outs(%66 : memref<64x32xf32>)
          loom.semaphore_give %64 : memref<16x32xf16>
          loom.semaphore_give %45 : memref<64x16xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %42 : memref<64x32xf32>, memref<64xf32>) outs(%67 : memref<64x32xf32>) {
          ^bb0(%in: f32, %in_6: f32, %out: f32):
            %93 = arith.mulf %in, %in_6 : f32
            linalg.yield %93 : f32
          }
          loom.semaphore_give %66 : memref<64x32xf32>
          loom.semaphore_give %42 : memref<64xf32>
          %68 = arith.addi %21, %c1 : index
          %69 = arith.muli %68, %c64 : index
          %70 = arith.ceildivui %69, %c64 : index
          %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
          %73 = loom.alloc [64] on @L1 : memref<64xf16>
          %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %76 = loom.alloc [64] on @L1 : memref<64xf32>
          %77 = loom.semaphore_take %76 : memref<64xf32> -> memref<64xf32>
          %78 = loom.alloc [64] on @L1 : memref<64xf32>
          %79 = loom.semaphore_take %78 : memref<64xf32> -> memref<64xf32>
          %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %70 step %c1 {
            %93 = arith.muli %arg14, %c64 : index
            %94 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %95 = arith.muli %arg12, %c262144 : index
            %96 = arith.addi %94, %95 : index
            %97 = arith.muli %43, %c65536 overflow<nsw> : index
            %98 = arith.addi %96, %97 : index
            %99 = arith.muli %21, %c16384 : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.addi %100, %93 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%101], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %36], LR : [%33, %38]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %102 = arith.addi %30, %93 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%50, %38]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%75 : memref<64xf16>) outs(%77 : memref<64xf32>) {
            ^bb0(%in: f16, %out: f32):
              %109 = arith.extf %in : f16 to f32
              linalg.yield %109 : f32
            }
            loom.semaphore_give %75 : memref<64xf16>
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%32, %36], LR : [%50, %38]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%74 : memref<64xf16>) outs(%79 : memref<64xf32>) {
            ^bb0(%in: f16, %out: f32):
              %109 = arith.extf %in : f16 to f32
              linalg.yield %109 : f32
            }
            loom.semaphore_give %74 : memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %40, %77, %79 : memref<64x64xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%72 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_10: f32, %in_11: f32, %in_12: f32, %out: f16):
              %109 = arith.truncf %cst_0 : f64 to f32
              %110 = arith.mulf %in_11, %109 : f32
              %111 = arith.mulf %in_10, %109 : f32
              %112 = arith.subf %111, %110 : f32
              %113 = math.powf %cst, %112 : f32
              %114 = arith.extf %in : f16 to f32
              %115 = arith.mulf %114, %113 : f32
              %116 = arith.mulf %115, %in_12 : f32
              %117 = arith.truncf %116 : f32 to f16
              linalg.yield %117 : f16
            }
            loom.semaphore_give %79 : memref<64xf32>
            loom.semaphore_give %77 : memref<64xf32>
            %103 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %104 = arith.muli %arg12, %c1048576 : index
            %105 = arith.addi %103, %104 : index
            %106 = arith.muli %arg8, %c512 : index
            %107 = arith.addi %105, %106 : index
            %108 = arith.addi %107, %51 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%108], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %62], LR : [%50, %62]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%72, %81 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf32>)
            loom.semaphore_give %81 : memref<64x32xf16>
            loom.semaphore_give %72 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %40 : memref<64xf32>
          %82 = loom.alloc [1] on @L1 : memref<f16>
          %83 = loom.semaphore_take %82 : memref<f16> -> memref<f16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
          %84 = arith.addi %35, %c3 : index
          loom.copy %reinterpret_cast_3, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %35], LR : [%c7, %84]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %85 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %86 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
          %87 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %88 = arith.muli %arg12, %c1048576 : index
          %89 = arith.addi %87, %88 : index
          %90 = arith.muli %arg8, %c512 : index
          %91 = arith.addi %89, %90 : index
          %92 = arith.addi %91, %51 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%32, %62], LR : [%50, %62]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %86, %83 : memref<64x32xf32>, memref<64x32xf16>, memref<f16>) outs(%86 : memref<64x32xf16>) {
          ^bb0(%in: f32, %in_6: f16, %in_7: f16, %out: f16):
            %93 = arith.extf %in_7 : f16 to f32
            %94 = arith.extf %in_6 : f16 to f32
            %95 = arith.mulf %94, %93 : f32
            %96 = arith.addf %in, %95 : f32
            %97 = arith.truncf %96 : f32 to f16
            linalg.yield %97 : f16
          }
          loom.semaphore_give %83 : memref<f16>
          loom.semaphore_give %67 : memref<64x32xf32>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %86, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%32, %62], LR : [%50, %62]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.semaphore_give %86 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c512 = arith.constant 512 : index
      %c8192 = arith.constant 8192 : index
      %c1024 = arith.constant 1024 : index
      %c16384 = arith.constant 16384 : index
      %c2097152 = arith.constant 2097152 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c131072 = arith.constant 131072 : index
      %c32768 = arith.constant 32768 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c4, %c2, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        %20 = arith.muli %arg8, %c8 : index
        %21 = arith.muli %arg9, %c64 : index
        %22 = loom.alloc [64] on @L1 : memref<64xf16>
        %23 = loom.semaphore_take %22 : memref<64xf16> -> memref<64xf16>
        %24 = arith.muli %arg11, %c32768 overflow<nsw> : index
        %25 = arith.muli %arg8, %c16384 : index
        %26 = arith.addi %24, %25 : index
        %27 = arith.muli %arg12, %c1024 : index
        %28 = arith.addi %26, %27 : index
        %29 = arith.addi %28, %21 : index
        %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %30 = arith.muli %arg11, %c4 : index
        %31 = arith.addi %arg9, %30 : index
        %32 = arith.muli %arg12, %c2 : index
        %33 = arith.muli %arg8, %c4 : index
        %34 = arith.addi %32, %33 : index
        %35 = arith.addi %32, %c1 : index
        %36 = arith.addi %35, %33 : index
        loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
        %37 = loom.alloc [64] on @L1 : memref<64xf32>
        %38 = loom.semaphore_take %37 : memref<64xf32> -> memref<64xf32>
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%23 : memref<64xf16>) outs(%38 : memref<64xf32>) {
        ^bb0(%in: f16, %out: f32):
          %91 = arith.extf %in : f16 to f32
          linalg.yield %91 : f32
        }
        loom.semaphore_give %23 : memref<64xf16>
        %39 = loom.alloc [64] on @L1 : memref<64xf32>
        %40 = loom.semaphore_take %39 : memref<64xf32> -> memref<64xf32>
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%38 : memref<64xf32>) outs(%40 : memref<64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %91 = arith.truncf %cst_0 : f64 to f32
          %92 = arith.mulf %in, %91 : f32
          %93 = math.powf %cst, %92 : f32
          linalg.yield %93 : f32
        }
        %41 = arith.divui %20, %c16 : index
        %42 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
        %43 = loom.semaphore_take %42 : memref<64x16xf16> -> memref<64x16xf16>
        %44 = arith.muli %arg12, %c16384 : index
        %45 = arith.addi %24, %44 : index
        %46 = arith.muli %41, %c16 overflow<nsw> : index
        %47 = arith.addi %45, %46 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
        %48 = arith.addi %30, %c3 : index
        loom.copy %reinterpret_cast_1, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%48, %36]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
        %49 = arith.muli %arg10, %c32 : index
        %50 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
        %51 = loom.semaphore_take %50 : memref<32x16xf16> -> memref<32x16xf16>
        %52 = arith.muli %arg11, %c131072 overflow<nsw> : index
        %53 = arith.muli %arg12, %c65536 : index
        %54 = arith.addi %52, %53 : index
        %55 = arith.muli %arg8, %c8192 : index
        %56 = arith.addi %54, %55 : index
        %57 = arith.muli %arg10, %c512 : index
        %58 = arith.addi %56, %57 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%58], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
        %59 = arith.addi %arg10, %32 : index
        %60 = arith.addi %59, %33 : index
        loom.copy %reinterpret_cast_2, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %60], LR : [%48, %60]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
        %61 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
        %62 = loom.semaphore_take %61 : memref<16x32xf16> -> memref<16x32xf16>
        linalg.transpose ins(%51 : memref<32x16xf16>) outs(%62 : memref<16x32xf16>) permutation = [1, 0] 
        loom.semaphore_give %51 : memref<32x16xf16>
        %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
        %64 = loom.semaphore_take %63 : memref<64x32xf32> -> memref<64x32xf32>
        %65 = loom.semaphore_take %63 : memref<64x32xf32> -> memref<64x32xf32>
        loom.matmul ins(%43, %62 : memref<64x16xf16>, memref<16x32xf16>) outs(%64 : memref<64x32xf32>)
        loom.semaphore_give %62 : memref<16x32xf16>
        loom.semaphore_give %43 : memref<64x16xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %40 : memref<64x32xf32>, memref<64xf32>) outs(%65 : memref<64x32xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %91 = arith.mulf %in, %in_6 : f32
          linalg.yield %91 : f32
        }
        loom.semaphore_give %64 : memref<64x32xf32>
        loom.semaphore_give %40 : memref<64xf32>
        %66 = arith.addi %arg9, %c1 : index
        %67 = arith.muli %66, %c64 : index
        %68 = arith.ceildivui %67, %c64 : index
        %69 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %70 = loom.semaphore_take %69 : memref<64x64xf16> -> memref<64x64xf16>
        %71 = loom.alloc [64] on @L1 : memref<64xf16>
        %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
        %73 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
        %74 = loom.alloc [64] on @L1 : memref<64xf32>
        %75 = loom.semaphore_take %74 : memref<64xf32> -> memref<64xf32>
        %76 = loom.alloc [64] on @L1 : memref<64xf32>
        %77 = loom.semaphore_take %76 : memref<64xf32> -> memref<64xf32>
        %78 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %79 = loom.semaphore_take %78 : memref<64x32xf16> -> memref<64x32xf16>
        scf.for %arg13 = %c0 to %68 step %c1 {
          %91 = arith.muli %arg13, %c64 : index
          %92 = arith.muli %arg11, %c524288 overflow<nsw> : index
          %93 = arith.muli %arg12, %c262144 : index
          %94 = arith.addi %92, %93 : index
          %95 = arith.muli %41, %c65536 overflow<nsw> : index
          %96 = arith.addi %94, %95 : index
          %97 = arith.muli %arg9, %c16384 : index
          %98 = arith.addi %96, %97 : index
          %99 = arith.addi %98, %91 : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%99], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_6, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
          %100 = arith.addi %28, %91 : index
          %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%100], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_7, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%48, %36]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%73 : memref<64xf16>) outs(%75 : memref<64xf32>) {
          ^bb0(%in: f16, %out: f32):
            %107 = arith.extf %in : f16 to f32
            linalg.yield %107 : f32
          }
          loom.semaphore_give %73 : memref<64xf16>
          %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%100], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_8, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%48, %36]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%72 : memref<64xf16>) outs(%77 : memref<64xf32>) {
          ^bb0(%in: f16, %out: f32):
            %107 = arith.extf %in : f16 to f32
            linalg.yield %107 : f32
          }
          loom.semaphore_give %72 : memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %38, %75, %77 : memref<64x64xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%70 : memref<64x64xf16>) {
          ^bb0(%in: f16, %in_10: f32, %in_11: f32, %in_12: f32, %out: f16):
            %107 = arith.truncf %cst_0 : f64 to f32
            %108 = arith.mulf %in_11, %107 : f32
            %109 = arith.mulf %in_10, %107 : f32
            %110 = arith.subf %109, %108 : f32
            %111 = math.powf %cst, %110 : f32
            %112 = arith.extf %in : f16 to f32
            %113 = arith.mulf %112, %111 : f32
            %114 = arith.mulf %113, %in_12 : f32
            %115 = arith.truncf %114 : f32 to f16
            linalg.yield %115 : f16
          }
          loom.semaphore_give %77 : memref<64xf32>
          loom.semaphore_give %75 : memref<64xf32>
          %101 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %102 = arith.muli %arg12, %c1048576 : index
          %103 = arith.addi %101, %102 : index
          %104 = arith.muli %arg8, %c512 : index
          %105 = arith.addi %103, %104 : index
          %106 = arith.addi %105, %49 : index
          %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%106], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %reinterpret_cast_9, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %60], LR : [%48, %60]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
          linalg.matmul ins(%70, %79 : memref<64x64xf16>, memref<64x32xf16>) outs(%65 : memref<64x32xf32>)
          loom.semaphore_give %79 : memref<64x32xf16>
          loom.semaphore_give %70 : memref<64x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        loom.semaphore_give %38 : memref<64xf32>
        %80 = loom.alloc [1] on @L1 : memref<f16>
        %81 = loom.semaphore_take %80 : memref<f16> -> memref<f16>
        %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%20], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
        %82 = arith.addi %33, %c3 : index
        loom.copy %reinterpret_cast_3, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %33], LR : [%c7, %82]) : memref<f16, strided<[], offset: ?>> to memref<f16>
        %83 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %84 = loom.semaphore_take %83 : memref<64x32xf16> -> memref<64x32xf16>
        %85 = arith.muli %arg11, %c2097152 overflow<nsw> : index
        %86 = arith.muli %arg12, %c1048576 : index
        %87 = arith.addi %85, %86 : index
        %88 = arith.muli %arg8, %c512 : index
        %89 = arith.addi %87, %88 : index
        %90 = arith.addi %89, %49 : index
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%90], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        loom.copy %reinterpret_cast_4, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %60], LR : [%48, %60]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %84, %81 : memref<64x32xf32>, memref<64x32xf16>, memref<f16>) outs(%84 : memref<64x32xf16>) {
        ^bb0(%in: f32, %in_6: f16, %in_7: f16, %out: f16):
          %91 = arith.extf %in_7 : f16 to f32
          %92 = arith.extf %in_6 : f16 to f32
          %93 = arith.mulf %92, %91 : f32
          %94 = arith.addf %in, %93 : f32
          %95 = arith.truncf %94 : f32 to f16
          linalg.yield %95 : f16
        }
        loom.semaphore_give %81 : memref<f16>
        loom.semaphore_give %65 : memref<64x32xf32>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%90], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        loom.copy %84, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%30, %60], LR : [%48, %60]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        loom.semaphore_give %84 : memref<64x32xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c512 = arith.constant 512 : index
      %c8192 = arith.constant 8192 : index
      %c1024 = arith.constant 1024 : index
      %c16384 = arith.constant 16384 : index
      %c2097152 = arith.constant 2097152 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c131072 = arith.constant 131072 : index
      %c32768 = arith.constant 32768 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c4, %c2, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        %20 = arith.muli %arg8, %c8 : index
        %21 = arith.muli %arg9, %c64 : index
        %22 = loom.alloc [64] on @L1 : memref<64xf16>
        %23 = loom.semaphore_take %22 : memref<64xf16> -> memref<64xf16>
        %24 = arith.muli %arg11, %c32768 overflow<nsw> : index
        %25 = arith.muli %arg8, %c16384 : index
        %26 = arith.addi %24, %25 : index
        %27 = arith.muli %arg12, %c1024 : index
        %28 = arith.addi %26, %27 : index
        %29 = arith.addi %28, %21 : index
        %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %30 = arith.muli %arg12, %c4 : index
        %31 = arith.addi %arg9, %30 : index
        %32 = arith.muli %arg11, %c2 : index
        %33 = arith.muli %arg8, %c4 : index
        %34 = arith.addi %32, %33 : index
        %35 = arith.addi %32, %c1 : index
        %36 = arith.addi %35, %33 : index
        loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
        %37 = loom.alloc [64] on @L1 : memref<64xf32>
        %38 = loom.semaphore_take %37 : memref<64xf32> -> memref<64xf32>
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%23 : memref<64xf16>) outs(%38 : memref<64xf32>) {
        ^bb0(%in: f16, %out: f32):
          %91 = arith.extf %in : f16 to f32
          linalg.yield %91 : f32
        }
        loom.semaphore_give %23 : memref<64xf16>
        %39 = loom.alloc [64] on @L1 : memref<64xf32>
        %40 = loom.semaphore_take %39 : memref<64xf32> -> memref<64xf32>
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%38 : memref<64xf32>) outs(%40 : memref<64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %91 = arith.truncf %cst_0 : f64 to f32
          %92 = arith.mulf %in, %91 : f32
          %93 = math.powf %cst, %92 : f32
          linalg.yield %93 : f32
        }
        %41 = arith.divui %20, %c16 : index
        %42 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
        %43 = loom.semaphore_take %42 : memref<64x16xf16> -> memref<64x16xf16>
        %44 = arith.muli %arg12, %c16384 : index
        %45 = arith.addi %24, %44 : index
        %46 = arith.muli %41, %c16 overflow<nsw> : index
        %47 = arith.addi %45, %46 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
        %48 = arith.addi %30, %c3 : index
        loom.copy %reinterpret_cast_1, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%48, %36]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
        %49 = arith.muli %arg10, %c32 : index
        %50 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
        %51 = loom.semaphore_take %50 : memref<32x16xf16> -> memref<32x16xf16>
        %52 = arith.muli %arg11, %c131072 overflow<nsw> : index
        %53 = arith.muli %arg12, %c65536 : index
        %54 = arith.addi %52, %53 : index
        %55 = arith.muli %arg8, %c8192 : index
        %56 = arith.addi %54, %55 : index
        %57 = arith.muli %arg10, %c512 : index
        %58 = arith.addi %56, %57 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%58], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
        %59 = arith.addi %arg10, %32 : index
        %60 = arith.addi %59, %33 : index
        loom.copy %reinterpret_cast_2, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %60], LR : [%48, %60]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
        %61 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
        %62 = loom.semaphore_take %61 : memref<16x32xf16> -> memref<16x32xf16>
        linalg.transpose ins(%51 : memref<32x16xf16>) outs(%62 : memref<16x32xf16>) permutation = [1, 0] 
        loom.semaphore_give %51 : memref<32x16xf16>
        %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
        %64 = loom.semaphore_take %63 : memref<64x32xf32> -> memref<64x32xf32>
        %65 = loom.semaphore_take %63 : memref<64x32xf32> -> memref<64x32xf32>
        loom.matmul ins(%43, %62 : memref<64x16xf16>, memref<16x32xf16>) outs(%64 : memref<64x32xf32>)
        loom.semaphore_give %62 : memref<16x32xf16>
        loom.semaphore_give %43 : memref<64x16xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %40 : memref<64x32xf32>, memref<64xf32>) outs(%65 : memref<64x32xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %91 = arith.mulf %in, %in_6 : f32
          linalg.yield %91 : f32
        }
        loom.semaphore_give %64 : memref<64x32xf32>
        loom.semaphore_give %40 : memref<64xf32>
        %66 = arith.addi %arg9, %c1 : index
        %67 = arith.muli %66, %c64 : index
        %68 = arith.ceildivui %67, %c64 : index
        %69 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %70 = loom.semaphore_take %69 : memref<64x64xf16> -> memref<64x64xf16>
        %71 = loom.alloc [64] on @L1 : memref<64xf16>
        %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
        %73 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
        %74 = loom.alloc [64] on @L1 : memref<64xf32>
        %75 = loom.semaphore_take %74 : memref<64xf32> -> memref<64xf32>
        %76 = loom.alloc [64] on @L1 : memref<64xf32>
        %77 = loom.semaphore_take %76 : memref<64xf32> -> memref<64xf32>
        %78 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %79 = loom.semaphore_take %78 : memref<64x32xf16> -> memref<64x32xf16>
        scf.for %arg13 = %c0 to %68 step %c1 {
          %91 = arith.muli %arg13, %c64 : index
          %92 = arith.muli %arg11, %c524288 overflow<nsw> : index
          %93 = arith.muli %arg12, %c262144 : index
          %94 = arith.addi %92, %93 : index
          %95 = arith.muli %41, %c65536 overflow<nsw> : index
          %96 = arith.addi %94, %95 : index
          %97 = arith.muli %arg9, %c16384 : index
          %98 = arith.addi %96, %97 : index
          %99 = arith.addi %98, %91 : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%99], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast_6, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
          %100 = arith.addi %28, %91 : index
          %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%100], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_7, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%48, %36]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%73 : memref<64xf16>) outs(%75 : memref<64xf32>) {
          ^bb0(%in: f16, %out: f32):
            %107 = arith.extf %in : f16 to f32
            linalg.yield %107 : f32
          }
          loom.semaphore_give %73 : memref<64xf16>
          %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%100], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
          loom.copy %reinterpret_cast_8, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%30, %34], LR : [%48, %36]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%72 : memref<64xf16>) outs(%77 : memref<64xf32>) {
          ^bb0(%in: f16, %out: f32):
            %107 = arith.extf %in : f16 to f32
            linalg.yield %107 : f32
          }
          loom.semaphore_give %72 : memref<64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %38, %75, %77 : memref<64x64xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%70 : memref<64x64xf16>) {
          ^bb0(%in: f16, %in_10: f32, %in_11: f32, %in_12: f32, %out: f16):
            %107 = arith.truncf %cst_0 : f64 to f32
            %108 = arith.mulf %in_11, %107 : f32
            %109 = arith.mulf %in_10, %107 : f32
            %110 = arith.subf %109, %108 : f32
            %111 = math.powf %cst, %110 : f32
            %112 = arith.extf %in : f16 to f32
            %113 = arith.mulf %112, %111 : f32
            %114 = arith.mulf %113, %in_12 : f32
            %115 = arith.truncf %114 : f32 to f16
            linalg.yield %115 : f16
          }
          loom.semaphore_give %77 : memref<64xf32>
          loom.semaphore_give %75 : memref<64xf32>
          %101 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %102 = arith.muli %arg12, %c1048576 : index
          %103 = arith.addi %101, %102 : index
          %104 = arith.muli %arg8, %c512 : index
          %105 = arith.addi %103, %104 : index
          %106 = arith.addi %105, %49 : index
          %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%106], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %reinterpret_cast_9, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %60], LR : [%48, %60]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
          linalg.matmul ins(%70, %79 : memref<64x64xf16>, memref<64x32xf16>) outs(%65 : memref<64x32xf32>)
          loom.semaphore_give %79 : memref<64x32xf16>
          loom.semaphore_give %70 : memref<64x64xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        loom.semaphore_give %38 : memref<64xf32>
        %80 = loom.alloc [1] on @L1 : memref<f16>
        %81 = loom.semaphore_take %80 : memref<f16> -> memref<f16>
        %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%20], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
        %82 = arith.addi %33, %c3 : index
        loom.copy %reinterpret_cast_3, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %33], LR : [%c7, %82]) : memref<f16, strided<[], offset: ?>> to memref<f16>
        %83 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %84 = loom.semaphore_take %83 : memref<64x32xf16> -> memref<64x32xf16>
        %85 = arith.muli %arg11, %c2097152 overflow<nsw> : index
        %86 = arith.muli %arg12, %c1048576 : index
        %87 = arith.addi %85, %86 : index
        %88 = arith.muli %arg8, %c512 : index
        %89 = arith.addi %87, %88 : index
        %90 = arith.addi %89, %49 : index
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%90], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        loom.copy %reinterpret_cast_4, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%30, %60], LR : [%48, %60]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %84, %81 : memref<64x32xf32>, memref<64x32xf16>, memref<f16>) outs(%84 : memref<64x32xf16>) {
        ^bb0(%in: f32, %in_6: f16, %in_7: f16, %out: f16):
          %91 = arith.extf %in_7 : f16 to f32
          %92 = arith.extf %in_6 : f16 to f32
          %93 = arith.mulf %92, %91 : f32
          %94 = arith.addf %in, %93 : f32
          %95 = arith.truncf %94 : f32 to f16
          linalg.yield %95 : f16
        }
        loom.semaphore_give %81 : memref<f16>
        loom.semaphore_give %65 : memref<64x32xf32>
        %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%90], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        loom.copy %84, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%30, %60], LR : [%48, %60]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        loom.semaphore_give %84 : memref<64x32xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c512 = arith.constant 512 : index
      %c8192 = arith.constant 8192 : index
      %c1024 = arith.constant 1024 : index
      %c16384 = arith.constant 16384 : index
      %c2097152 = arith.constant 2097152 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c131072 = arith.constant 131072 : index
      %c32768 = arith.constant 32768 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c2, %c4) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg9, %20 : index
          %22 = arith.muli %arg8, %c8 : index
          %23 = arith.muli %21, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c32768 overflow<nsw> : index
          %27 = arith.muli %arg8, %c16384 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg11, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg8, %c4 : index
          %35 = arith.addi %33, %34 : index
          %36 = arith.muli %arg12, %c2 : index
          %37 = arith.addi %36, %c1 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %36], LR : [%35, %37]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
          %38 = loom.alloc [64] on @L1 : memref<64xf32>
          %39 = loom.semaphore_take %38 : memref<64xf32> -> memref<64xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25 : memref<64xf16>) outs(%39 : memref<64xf32>) {
          ^bb0(%in: f16, %out: f32):
            %93 = arith.extf %in : f16 to f32
            linalg.yield %93 : f32
          }
          loom.semaphore_give %25 : memref<64xf16>
          %40 = loom.alloc [64] on @L1 : memref<64xf32>
          %41 = loom.semaphore_take %40 : memref<64xf32> -> memref<64xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%39 : memref<64xf32>) outs(%41 : memref<64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %93 = arith.truncf %cst_0 : f64 to f32
            %94 = arith.mulf %in, %93 : f32
            %95 = math.powf %cst, %94 : f32
            linalg.yield %95 : f32
          }
          %42 = arith.divui %22, %c16 : index
          %43 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
          %44 = loom.semaphore_take %43 : memref<64x16xf16> -> memref<64x16xf16>
          %45 = arith.muli %arg12, %c16384 : index
          %46 = arith.addi %26, %45 : index
          %47 = arith.muli %42, %c16 overflow<nsw> : index
          %48 = arith.addi %46, %47 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
          %49 = arith.addi %32, %34 : index
          %50 = arith.addi %32, %c1 : index
          %51 = arith.addi %50, %34 : index
          loom.copy %reinterpret_cast_1, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%49, %36], LR : [%51, %37]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
          %52 = arith.muli %arg10, %c32 : index
          %53 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
          %54 = loom.semaphore_take %53 : memref<32x16xf16> -> memref<32x16xf16>
          %55 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %56 = arith.muli %arg12, %c65536 : index
          %57 = arith.addi %55, %56 : index
          %58 = arith.muli %arg8, %c8192 : index
          %59 = arith.addi %57, %58 : index
          %60 = arith.muli %arg10, %c512 : index
          %61 = arith.addi %59, %60 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%61], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
          %62 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_2, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
          %63 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
          %64 = loom.semaphore_take %63 : memref<16x32xf16> -> memref<16x32xf16>
          linalg.transpose ins(%54 : memref<32x16xf16>) outs(%64 : memref<16x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %54 : memref<32x16xf16>
          %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
          %66 = loom.semaphore_take %65 : memref<64x32xf32> -> memref<64x32xf32>
          %67 = loom.semaphore_take %65 : memref<64x32xf32> -> memref<64x32xf32>
          loom.matmul ins(%44, %64 : memref<64x16xf16>, memref<16x32xf16>) outs(%66 : memref<64x32xf32>)
          loom.semaphore_give %64 : memref<16x32xf16>
          loom.semaphore_give %44 : memref<64x16xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %41 : memref<64x32xf32>, memref<64xf32>) outs(%67 : memref<64x32xf32>) {
          ^bb0(%in: f32, %in_6: f32, %out: f32):
            %93 = arith.mulf %in, %in_6 : f32
            linalg.yield %93 : f32
          }
          loom.semaphore_give %66 : memref<64x32xf32>
          loom.semaphore_give %41 : memref<64xf32>
          %68 = arith.addi %21, %c1 : index
          %69 = arith.muli %68, %c64 : index
          %70 = arith.ceildivui %69, %c64 : index
          %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
          %73 = loom.alloc [64] on @L1 : memref<64xf16>
          %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %76 = loom.alloc [64] on @L1 : memref<64xf32>
          %77 = loom.semaphore_take %76 : memref<64xf32> -> memref<64xf32>
          %78 = loom.alloc [64] on @L1 : memref<64xf32>
          %79 = loom.semaphore_take %78 : memref<64xf32> -> memref<64xf32>
          %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %70 step %c1 {
            %93 = arith.muli %arg14, %c64 : index
            %94 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %95 = arith.muli %arg12, %c262144 : index
            %96 = arith.addi %94, %95 : index
            %97 = arith.muli %42, %c65536 overflow<nsw> : index
            %98 = arith.addi %96, %97 : index
            %99 = arith.muli %21, %c16384 : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.addi %100, %93 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%101], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %36], LR : [%35, %37]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %102 = arith.addi %30, %93 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%49, %36], LR : [%51, %37]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%75 : memref<64xf16>) outs(%77 : memref<64xf32>) {
            ^bb0(%in: f16, %out: f32):
              %109 = arith.extf %in : f16 to f32
              linalg.yield %109 : f32
            }
            loom.semaphore_give %75 : memref<64xf16>
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%49, %36], LR : [%51, %37]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%74 : memref<64xf16>) outs(%79 : memref<64xf32>) {
            ^bb0(%in: f16, %out: f32):
              %109 = arith.extf %in : f16 to f32
              linalg.yield %109 : f32
            }
            loom.semaphore_give %74 : memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %39, %77, %79 : memref<64x64xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%72 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_10: f32, %in_11: f32, %in_12: f32, %out: f16):
              %109 = arith.truncf %cst_0 : f64 to f32
              %110 = arith.mulf %in_11, %109 : f32
              %111 = arith.mulf %in_10, %109 : f32
              %112 = arith.subf %111, %110 : f32
              %113 = math.powf %cst, %112 : f32
              %114 = arith.extf %in : f16 to f32
              %115 = arith.mulf %114, %113 : f32
              %116 = arith.mulf %115, %in_12 : f32
              %117 = arith.truncf %116 : f32 to f16
              linalg.yield %117 : f16
            }
            loom.semaphore_give %79 : memref<64xf32>
            loom.semaphore_give %77 : memref<64xf32>
            %103 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %104 = arith.muli %arg12, %c1048576 : index
            %105 = arith.addi %103, %104 : index
            %106 = arith.muli %arg8, %c512 : index
            %107 = arith.addi %105, %106 : index
            %108 = arith.addi %107, %52 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%108], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%72, %81 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf32>)
            loom.semaphore_give %81 : memref<64x32xf16>
            loom.semaphore_give %72 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %39 : memref<64xf32>
          %82 = loom.alloc [1] on @L1 : memref<f16>
          %83 = loom.semaphore_take %82 : memref<f16> -> memref<f16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
          %84 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_3, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%84, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %85 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %86 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
          %87 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %88 = arith.muli %arg12, %c1048576 : index
          %89 = arith.addi %87, %88 : index
          %90 = arith.muli %arg8, %c512 : index
          %91 = arith.addi %89, %90 : index
          %92 = arith.addi %91, %52 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %86, %83 : memref<64x32xf32>, memref<64x32xf16>, memref<f16>) outs(%86 : memref<64x32xf16>) {
          ^bb0(%in: f32, %in_6: f16, %in_7: f16, %out: f16):
            %93 = arith.extf %in_7 : f16 to f32
            %94 = arith.extf %in_6 : f16 to f32
            %95 = arith.mulf %94, %93 : f32
            %96 = arith.addf %in, %95 : f32
            %97 = arith.truncf %96 : f32 to f16
            linalg.yield %97 : f16
          }
          loom.semaphore_give %83 : memref<f16>
          loom.semaphore_give %67 : memref<64x32xf32>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %86, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.semaphore_give %86 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c512 = arith.constant 512 : index
      %c8192 = arith.constant 8192 : index
      %c1024 = arith.constant 1024 : index
      %c16384 = arith.constant 16384 : index
      %c2097152 = arith.constant 2097152 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c131072 = arith.constant 131072 : index
      %c32768 = arith.constant 32768 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c4, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg9, %20 : index
          %22 = arith.muli %arg8, %c8 : index
          %23 = arith.muli %21, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c32768 overflow<nsw> : index
          %27 = arith.muli %arg8, %c16384 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg12, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg8, %c4 : index
          %35 = arith.addi %33, %34 : index
          %36 = arith.muli %arg11, %c2 : index
          %37 = arith.addi %36, %c1 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %36], LR : [%35, %37]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
          %38 = loom.alloc [64] on @L1 : memref<64xf32>
          %39 = loom.semaphore_take %38 : memref<64xf32> -> memref<64xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25 : memref<64xf16>) outs(%39 : memref<64xf32>) {
          ^bb0(%in: f16, %out: f32):
            %93 = arith.extf %in : f16 to f32
            linalg.yield %93 : f32
          }
          loom.semaphore_give %25 : memref<64xf16>
          %40 = loom.alloc [64] on @L1 : memref<64xf32>
          %41 = loom.semaphore_take %40 : memref<64xf32> -> memref<64xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%39 : memref<64xf32>) outs(%41 : memref<64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %93 = arith.truncf %cst_0 : f64 to f32
            %94 = arith.mulf %in, %93 : f32
            %95 = math.powf %cst, %94 : f32
            linalg.yield %95 : f32
          }
          %42 = arith.divui %22, %c16 : index
          %43 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
          %44 = loom.semaphore_take %43 : memref<64x16xf16> -> memref<64x16xf16>
          %45 = arith.muli %arg12, %c16384 : index
          %46 = arith.addi %26, %45 : index
          %47 = arith.muli %42, %c16 overflow<nsw> : index
          %48 = arith.addi %46, %47 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
          %49 = arith.addi %32, %34 : index
          %50 = arith.addi %32, %c1 : index
          %51 = arith.addi %50, %34 : index
          loom.copy %reinterpret_cast_1, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%49, %36], LR : [%51, %37]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
          %52 = arith.muli %arg10, %c32 : index
          %53 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
          %54 = loom.semaphore_take %53 : memref<32x16xf16> -> memref<32x16xf16>
          %55 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %56 = arith.muli %arg12, %c65536 : index
          %57 = arith.addi %55, %56 : index
          %58 = arith.muli %arg8, %c8192 : index
          %59 = arith.addi %57, %58 : index
          %60 = arith.muli %arg10, %c512 : index
          %61 = arith.addi %59, %60 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%61], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
          %62 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_2, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
          %63 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
          %64 = loom.semaphore_take %63 : memref<16x32xf16> -> memref<16x32xf16>
          linalg.transpose ins(%54 : memref<32x16xf16>) outs(%64 : memref<16x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %54 : memref<32x16xf16>
          %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
          %66 = loom.semaphore_take %65 : memref<64x32xf32> -> memref<64x32xf32>
          %67 = loom.semaphore_take %65 : memref<64x32xf32> -> memref<64x32xf32>
          loom.matmul ins(%44, %64 : memref<64x16xf16>, memref<16x32xf16>) outs(%66 : memref<64x32xf32>)
          loom.semaphore_give %64 : memref<16x32xf16>
          loom.semaphore_give %44 : memref<64x16xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %41 : memref<64x32xf32>, memref<64xf32>) outs(%67 : memref<64x32xf32>) {
          ^bb0(%in: f32, %in_6: f32, %out: f32):
            %93 = arith.mulf %in, %in_6 : f32
            linalg.yield %93 : f32
          }
          loom.semaphore_give %66 : memref<64x32xf32>
          loom.semaphore_give %41 : memref<64xf32>
          %68 = arith.addi %21, %c1 : index
          %69 = arith.muli %68, %c64 : index
          %70 = arith.ceildivui %69, %c64 : index
          %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
          %73 = loom.alloc [64] on @L1 : memref<64xf16>
          %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %76 = loom.alloc [64] on @L1 : memref<64xf32>
          %77 = loom.semaphore_take %76 : memref<64xf32> -> memref<64xf32>
          %78 = loom.alloc [64] on @L1 : memref<64xf32>
          %79 = loom.semaphore_take %78 : memref<64xf32> -> memref<64xf32>
          %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %70 step %c1 {
            %93 = arith.muli %arg14, %c64 : index
            %94 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %95 = arith.muli %arg12, %c262144 : index
            %96 = arith.addi %94, %95 : index
            %97 = arith.muli %42, %c65536 overflow<nsw> : index
            %98 = arith.addi %96, %97 : index
            %99 = arith.muli %21, %c16384 : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.addi %100, %93 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%101], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%35, %36], LR : [%35, %37]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %102 = arith.addi %30, %93 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%49, %36], LR : [%51, %37]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%75 : memref<64xf16>) outs(%77 : memref<64xf32>) {
            ^bb0(%in: f16, %out: f32):
              %109 = arith.extf %in : f16 to f32
              linalg.yield %109 : f32
            }
            loom.semaphore_give %75 : memref<64xf16>
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%49, %36], LR : [%51, %37]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%74 : memref<64xf16>) outs(%79 : memref<64xf32>) {
            ^bb0(%in: f16, %out: f32):
              %109 = arith.extf %in : f16 to f32
              linalg.yield %109 : f32
            }
            loom.semaphore_give %74 : memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %39, %77, %79 : memref<64x64xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%72 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_10: f32, %in_11: f32, %in_12: f32, %out: f16):
              %109 = arith.truncf %cst_0 : f64 to f32
              %110 = arith.mulf %in_11, %109 : f32
              %111 = arith.mulf %in_10, %109 : f32
              %112 = arith.subf %111, %110 : f32
              %113 = math.powf %cst, %112 : f32
              %114 = arith.extf %in : f16 to f32
              %115 = arith.mulf %114, %113 : f32
              %116 = arith.mulf %115, %in_12 : f32
              %117 = arith.truncf %116 : f32 to f16
              linalg.yield %117 : f16
            }
            loom.semaphore_give %79 : memref<64xf32>
            loom.semaphore_give %77 : memref<64xf32>
            %103 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %104 = arith.muli %arg12, %c1048576 : index
            %105 = arith.addi %103, %104 : index
            %106 = arith.muli %arg8, %c512 : index
            %107 = arith.addi %105, %106 : index
            %108 = arith.addi %107, %52 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%108], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%72, %81 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf32>)
            loom.semaphore_give %81 : memref<64x32xf16>
            loom.semaphore_give %72 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %39 : memref<64xf32>
          %82 = loom.alloc [1] on @L1 : memref<f16>
          %83 = loom.semaphore_take %82 : memref<f16> -> memref<f16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
          %84 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_3, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%84, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %85 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %86 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
          %87 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %88 = arith.muli %arg12, %c1048576 : index
          %89 = arith.addi %87, %88 : index
          %90 = arith.muli %arg8, %c512 : index
          %91 = arith.addi %89, %90 : index
          %92 = arith.addi %91, %52 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %86, %83 : memref<64x32xf32>, memref<64x32xf16>, memref<f16>) outs(%86 : memref<64x32xf16>) {
          ^bb0(%in: f32, %in_6: f16, %in_7: f16, %out: f16):
            %93 = arith.extf %in_7 : f16 to f32
            %94 = arith.extf %in_6 : f16 to f32
            %95 = arith.mulf %94, %93 : f32
            %96 = arith.addf %in, %95 : f32
            %97 = arith.truncf %96 : f32 to f16
            linalg.yield %97 : f16
          }
          loom.semaphore_give %83 : memref<f16>
          loom.semaphore_give %67 : memref<64x32xf32>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %86, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.semaphore_give %86 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c512 = arith.constant 512 : index
      %c8192 = arith.constant 8192 : index
      %c1024 = arith.constant 1024 : index
      %c16384 = arith.constant 16384 : index
      %c2097152 = arith.constant 2097152 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c131072 = arith.constant 131072 : index
      %c32768 = arith.constant 32768 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c4, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg9, %20 : index
          %22 = arith.muli %arg8, %c8 : index
          %23 = arith.muli %21, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c32768 overflow<nsw> : index
          %27 = arith.muli %arg8, %c16384 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg11, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg8, %c4 : index
          %35 = arith.addi %33, %34 : index
          %36 = arith.muli %arg12, %c4 : index
          %37 = arith.addi %36, %c3 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%35, %36], LR : [%35, %37]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
          %38 = loom.alloc [64] on @L1 : memref<64xf32>
          %39 = loom.semaphore_take %38 : memref<64xf32> -> memref<64xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25 : memref<64xf16>) outs(%39 : memref<64xf32>) {
          ^bb0(%in: f16, %out: f32):
            %93 = arith.extf %in : f16 to f32
            linalg.yield %93 : f32
          }
          loom.semaphore_give %25 : memref<64xf16>
          %40 = loom.alloc [64] on @L1 : memref<64xf32>
          %41 = loom.semaphore_take %40 : memref<64xf32> -> memref<64xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%39 : memref<64xf32>) outs(%41 : memref<64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %93 = arith.truncf %cst_0 : f64 to f32
            %94 = arith.mulf %in, %93 : f32
            %95 = math.powf %cst, %94 : f32
            linalg.yield %95 : f32
          }
          %42 = arith.divui %22, %c16 : index
          %43 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
          %44 = loom.semaphore_take %43 : memref<64x16xf16> -> memref<64x16xf16>
          %45 = arith.muli %arg12, %c16384 : index
          %46 = arith.addi %26, %45 : index
          %47 = arith.muli %42, %c16 overflow<nsw> : index
          %48 = arith.addi %46, %47 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
          %49 = arith.addi %32, %34 : index
          %50 = arith.addi %32, %c1 : index
          %51 = arith.addi %50, %34 : index
          loom.copy %reinterpret_cast_1, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%49, %36], LR : [%51, %37]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
          %52 = arith.muli %arg10, %c32 : index
          %53 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
          %54 = loom.semaphore_take %53 : memref<32x16xf16> -> memref<32x16xf16>
          %55 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %56 = arith.muli %arg12, %c65536 : index
          %57 = arith.addi %55, %56 : index
          %58 = arith.muli %arg8, %c8192 : index
          %59 = arith.addi %57, %58 : index
          %60 = arith.muli %arg10, %c512 : index
          %61 = arith.addi %59, %60 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%61], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
          %62 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_2, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
          %63 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
          %64 = loom.semaphore_take %63 : memref<16x32xf16> -> memref<16x32xf16>
          linalg.transpose ins(%54 : memref<32x16xf16>) outs(%64 : memref<16x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %54 : memref<32x16xf16>
          %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
          %66 = loom.semaphore_take %65 : memref<64x32xf32> -> memref<64x32xf32>
          %67 = loom.semaphore_take %65 : memref<64x32xf32> -> memref<64x32xf32>
          loom.matmul ins(%44, %64 : memref<64x16xf16>, memref<16x32xf16>) outs(%66 : memref<64x32xf32>)
          loom.semaphore_give %64 : memref<16x32xf16>
          loom.semaphore_give %44 : memref<64x16xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %41 : memref<64x32xf32>, memref<64xf32>) outs(%67 : memref<64x32xf32>) {
          ^bb0(%in: f32, %in_6: f32, %out: f32):
            %93 = arith.mulf %in, %in_6 : f32
            linalg.yield %93 : f32
          }
          loom.semaphore_give %66 : memref<64x32xf32>
          loom.semaphore_give %41 : memref<64xf32>
          %68 = arith.addi %21, %c1 : index
          %69 = arith.muli %68, %c64 : index
          %70 = arith.ceildivui %69, %c64 : index
          %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
          %73 = loom.alloc [64] on @L1 : memref<64xf16>
          %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %76 = loom.alloc [64] on @L1 : memref<64xf32>
          %77 = loom.semaphore_take %76 : memref<64xf32> -> memref<64xf32>
          %78 = loom.alloc [64] on @L1 : memref<64xf32>
          %79 = loom.semaphore_take %78 : memref<64xf32> -> memref<64xf32>
          %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %70 step %c1 {
            %93 = arith.muli %arg14, %c64 : index
            %94 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %95 = arith.muli %arg12, %c262144 : index
            %96 = arith.addi %94, %95 : index
            %97 = arith.muli %42, %c65536 overflow<nsw> : index
            %98 = arith.addi %96, %97 : index
            %99 = arith.muli %21, %c16384 : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.addi %100, %93 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%101], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%35, %36], LR : [%35, %37]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %102 = arith.addi %30, %93 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%49, %36], LR : [%51, %37]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%75 : memref<64xf16>) outs(%77 : memref<64xf32>) {
            ^bb0(%in: f16, %out: f32):
              %109 = arith.extf %in : f16 to f32
              linalg.yield %109 : f32
            }
            loom.semaphore_give %75 : memref<64xf16>
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%49, %36], LR : [%51, %37]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%74 : memref<64xf16>) outs(%79 : memref<64xf32>) {
            ^bb0(%in: f16, %out: f32):
              %109 = arith.extf %in : f16 to f32
              linalg.yield %109 : f32
            }
            loom.semaphore_give %74 : memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %39, %77, %79 : memref<64x64xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%72 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_10: f32, %in_11: f32, %in_12: f32, %out: f16):
              %109 = arith.truncf %cst_0 : f64 to f32
              %110 = arith.mulf %in_11, %109 : f32
              %111 = arith.mulf %in_10, %109 : f32
              %112 = arith.subf %111, %110 : f32
              %113 = math.powf %cst, %112 : f32
              %114 = arith.extf %in : f16 to f32
              %115 = arith.mulf %114, %113 : f32
              %116 = arith.mulf %115, %in_12 : f32
              %117 = arith.truncf %116 : f32 to f16
              linalg.yield %117 : f16
            }
            loom.semaphore_give %79 : memref<64xf32>
            loom.semaphore_give %77 : memref<64xf32>
            %103 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %104 = arith.muli %arg12, %c1048576 : index
            %105 = arith.addi %103, %104 : index
            %106 = arith.muli %arg8, %c512 : index
            %107 = arith.addi %105, %106 : index
            %108 = arith.addi %107, %52 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%108], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%72, %81 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf32>)
            loom.semaphore_give %81 : memref<64x32xf16>
            loom.semaphore_give %72 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %39 : memref<64xf32>
          %82 = loom.alloc [1] on @L1 : memref<f16>
          %83 = loom.semaphore_take %82 : memref<f16> -> memref<f16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
          %84 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_3, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%84, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %85 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %86 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
          %87 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %88 = arith.muli %arg12, %c1048576 : index
          %89 = arith.addi %87, %88 : index
          %90 = arith.muli %arg8, %c512 : index
          %91 = arith.addi %89, %90 : index
          %92 = arith.addi %91, %52 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %86, %83 : memref<64x32xf32>, memref<64x32xf16>, memref<f16>) outs(%86 : memref<64x32xf16>) {
          ^bb0(%in: f32, %in_6: f16, %in_7: f16, %out: f16):
            %93 = arith.extf %in_7 : f16 to f32
            %94 = arith.extf %in_6 : f16 to f32
            %95 = arith.mulf %94, %93 : f32
            %96 = arith.addf %in, %95 : f32
            %97 = arith.truncf %96 : f32 to f16
            linalg.yield %97 : f16
          }
          loom.semaphore_give %83 : memref<f16>
          loom.semaphore_give %67 : memref<64x32xf32>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %86, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.semaphore_give %86 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c512 = arith.constant 512 : index
      %c8192 = arith.constant 8192 : index
      %c1024 = arith.constant 1024 : index
      %c16384 = arith.constant 16384 : index
      %c2097152 = arith.constant 2097152 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c131072 = arith.constant 131072 : index
      %c32768 = arith.constant 32768 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c4, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg9, %20 : index
          %22 = arith.muli %arg8, %c8 : index
          %23 = arith.muli %21, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = arith.muli %arg11, %c32768 overflow<nsw> : index
          %27 = arith.muli %arg8, %c16384 : index
          %28 = arith.addi %26, %27 : index
          %29 = arith.muli %arg12, %c1024 : index
          %30 = arith.addi %28, %29 : index
          %31 = arith.addi %30, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
          %32 = arith.muli %arg12, %c2 : index
          %33 = arith.addi %arg9, %32 : index
          %34 = arith.muli %arg8, %c4 : index
          %35 = arith.addi %33, %34 : index
          %36 = arith.muli %arg11, %c4 : index
          %37 = arith.addi %36, %c3 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%35, %36], LR : [%35, %37]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
          %38 = loom.alloc [64] on @L1 : memref<64xf32>
          %39 = loom.semaphore_take %38 : memref<64xf32> -> memref<64xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25 : memref<64xf16>) outs(%39 : memref<64xf32>) {
          ^bb0(%in: f16, %out: f32):
            %93 = arith.extf %in : f16 to f32
            linalg.yield %93 : f32
          }
          loom.semaphore_give %25 : memref<64xf16>
          %40 = loom.alloc [64] on @L1 : memref<64xf32>
          %41 = loom.semaphore_take %40 : memref<64xf32> -> memref<64xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%39 : memref<64xf32>) outs(%41 : memref<64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %93 = arith.truncf %cst_0 : f64 to f32
            %94 = arith.mulf %in, %93 : f32
            %95 = math.powf %cst, %94 : f32
            linalg.yield %95 : f32
          }
          %42 = arith.divui %22, %c16 : index
          %43 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
          %44 = loom.semaphore_take %43 : memref<64x16xf16> -> memref<64x16xf16>
          %45 = arith.muli %arg12, %c16384 : index
          %46 = arith.addi %26, %45 : index
          %47 = arith.muli %42, %c16 overflow<nsw> : index
          %48 = arith.addi %46, %47 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
          %49 = arith.addi %32, %34 : index
          %50 = arith.addi %32, %c1 : index
          %51 = arith.addi %50, %34 : index
          loom.copy %reinterpret_cast_1, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%49, %36], LR : [%51, %37]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
          %52 = arith.muli %arg10, %c32 : index
          %53 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
          %54 = loom.semaphore_take %53 : memref<32x16xf16> -> memref<32x16xf16>
          %55 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %56 = arith.muli %arg12, %c65536 : index
          %57 = arith.addi %55, %56 : index
          %58 = arith.muli %arg8, %c8192 : index
          %59 = arith.addi %57, %58 : index
          %60 = arith.muli %arg10, %c512 : index
          %61 = arith.addi %59, %60 : index
          %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%61], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
          %62 = arith.addi %arg10, %36 : index
          loom.copy %reinterpret_cast_2, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
          %63 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
          %64 = loom.semaphore_take %63 : memref<16x32xf16> -> memref<16x32xf16>
          linalg.transpose ins(%54 : memref<32x16xf16>) outs(%64 : memref<16x32xf16>) permutation = [1, 0] 
          loom.semaphore_give %54 : memref<32x16xf16>
          %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
          %66 = loom.semaphore_take %65 : memref<64x32xf32> -> memref<64x32xf32>
          %67 = loom.semaphore_take %65 : memref<64x32xf32> -> memref<64x32xf32>
          loom.matmul ins(%44, %64 : memref<64x16xf16>, memref<16x32xf16>) outs(%66 : memref<64x32xf32>)
          loom.semaphore_give %64 : memref<16x32xf16>
          loom.semaphore_give %44 : memref<64x16xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %41 : memref<64x32xf32>, memref<64xf32>) outs(%67 : memref<64x32xf32>) {
          ^bb0(%in: f32, %in_6: f32, %out: f32):
            %93 = arith.mulf %in, %in_6 : f32
            linalg.yield %93 : f32
          }
          loom.semaphore_give %66 : memref<64x32xf32>
          loom.semaphore_give %41 : memref<64xf32>
          %68 = arith.addi %21, %c1 : index
          %69 = arith.muli %68, %c64 : index
          %70 = arith.ceildivui %69, %c64 : index
          %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
          %73 = loom.alloc [64] on @L1 : memref<64xf16>
          %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
          %76 = loom.alloc [64] on @L1 : memref<64xf32>
          %77 = loom.semaphore_take %76 : memref<64xf32> -> memref<64xf32>
          %78 = loom.alloc [64] on @L1 : memref<64xf32>
          %79 = loom.semaphore_take %78 : memref<64xf32> -> memref<64xf32>
          %80 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %81 = loom.semaphore_take %80 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %70 step %c1 {
            %93 = arith.muli %arg14, %c64 : index
            %94 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %95 = arith.muli %arg12, %c262144 : index
            %96 = arith.addi %94, %95 : index
            %97 = arith.muli %42, %c65536 overflow<nsw> : index
            %98 = arith.addi %96, %97 : index
            %99 = arith.muli %21, %c16384 : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.addi %100, %93 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%101], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%35, %36], LR : [%35, %37]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            %102 = arith.addi %30, %93 : index
            %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%49, %36], LR : [%51, %37]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%75 : memref<64xf16>) outs(%77 : memref<64xf32>) {
            ^bb0(%in: f16, %out: f32):
              %109 = arith.extf %in : f16 to f32
              linalg.yield %109 : f32
            }
            loom.semaphore_give %75 : memref<64xf16>
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_8, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%49, %36], LR : [%51, %37]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%74 : memref<64xf16>) outs(%79 : memref<64xf32>) {
            ^bb0(%in: f16, %out: f32):
              %109 = arith.extf %in : f16 to f32
              linalg.yield %109 : f32
            }
            loom.semaphore_give %74 : memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %39, %77, %79 : memref<64x64xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>) outs(%72 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_10: f32, %in_11: f32, %in_12: f32, %out: f16):
              %109 = arith.truncf %cst_0 : f64 to f32
              %110 = arith.mulf %in_11, %109 : f32
              %111 = arith.mulf %in_10, %109 : f32
              %112 = arith.subf %111, %110 : f32
              %113 = math.powf %cst, %112 : f32
              %114 = arith.extf %in : f16 to f32
              %115 = arith.mulf %114, %113 : f32
              %116 = arith.mulf %115, %in_12 : f32
              %117 = arith.truncf %116 : f32 to f16
              linalg.yield %117 : f16
            }
            loom.semaphore_give %79 : memref<64xf32>
            loom.semaphore_give %77 : memref<64xf32>
            %103 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %104 = arith.muli %arg12, %c1048576 : index
            %105 = arith.addi %103, %104 : index
            %106 = arith.muli %arg8, %c512 : index
            %107 = arith.addi %105, %106 : index
            %108 = arith.addi %107, %52 : index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%108], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
            linalg.matmul ins(%72, %81 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf32>)
            loom.semaphore_give %81 : memref<64x32xf16>
            loom.semaphore_give %72 : memref<64x64xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %39 : memref<64xf32>
          %82 = loom.alloc [1] on @L1 : memref<f16>
          %83 = loom.semaphore_take %82 : memref<f16> -> memref<f16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
          %84 = arith.addi %34, %c3 : index
          loom.copy %reinterpret_cast_3, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%34, %c0], LR : [%84, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          %85 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %86 = loom.semaphore_take %85 : memref<64x32xf16> -> memref<64x32xf16>
          %87 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %88 = arith.muli %arg12, %c1048576 : index
          %89 = arith.addi %87, %88 : index
          %90 = arith.muli %arg8, %c512 : index
          %91 = arith.addi %89, %90 : index
          %92 = arith.addi %91, %52 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %86, %83 : memref<64x32xf32>, memref<64x32xf16>, memref<f16>) outs(%86 : memref<64x32xf16>) {
          ^bb0(%in: f32, %in_6: f16, %in_7: f16, %out: f16):
            %93 = arith.extf %in_7 : f16 to f32
            %94 = arith.extf %in_6 : f16 to f32
            %95 = arith.mulf %94, %93 : f32
            %96 = arith.addf %in, %95 : f32
            %97 = arith.truncf %96 : f32 to f16
            linalg.yield %97 : f16
          }
          loom.semaphore_give %83 : memref<f16>
          loom.semaphore_give %67 : memref<64x32xf32>
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%92], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.copy %86, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%49, %62], LR : [%51, %62]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
          loom.semaphore_give %86 : memref<64x32xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
}
