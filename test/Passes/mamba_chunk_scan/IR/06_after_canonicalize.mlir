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
      %c1024 = arith.constant 1024 : index
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
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (4) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg13)
                  %21 = arith.muli %arg8, %c32 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %20, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg11, %c2 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg12, %c2 : index
                  %30 = arith.muli %arg8, %c4 : index
                  %31 = arith.addi %29, %30 : index
                  %32 = arith.addi %29, %c1 : index
                  %33 = arith.addi %32, %30 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%28, %31], LR : [%28, %33]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %34 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %35 = arith.muli %arg12, %c1024 : index
                  %36 = arith.divui %21, %c64 : index
                  %37 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %38 = loom.semaphore_take %37 : memref<64x64xf16> -> memref<64x64xf16>
                  %c0_2 = arith.constant 0 : index
                  %39 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %35, %36, %c0_2)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%39], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                  %40 = arith.addi %27, %c1 : index
                  loom.copy %reinterpret_cast_3, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%27, %31], LR : [%40, %33]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                  %41 = loom.bufferize_to_tensor %38[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %42 = arith.muli %arg10, %c32 : index
                  %43 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %44 = loom.semaphore_take %43 : memref<32x64xf16> -> memref<32x64xf16>
                  %c0_4 = arith.constant 0 : index
                  %45 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %22, %21, %42, %c0_4)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%45], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
                  %46 = arith.addi %arg10, %29 : index
                  %47 = arith.addi %46, %30 : index
                  loom.copy %reinterpret_cast_5, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%27, %47], LR : [%40, %47]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
                  %48 = loom.bufferize_to_tensor %44[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %49 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %50 = loom.semaphore_take %49 : memref<64x32xf16> -> memref<64x32xf16>
                  %51 = loom.init_tensor %50[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %transposed = linalg.transpose ins(%48 : tensor<32x64xf16>) outs(%51 : tensor<64x32xf16>) permutation = [1, 0] 
                  loom.semaphore_give %44 : memref<32x64xf16>
                  %52 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %53 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
                  %54 = loom.init_tensor %53[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %55 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
                  %56 = loom.init_tensor %55[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %57 = linalg.fill ins(%cst_0 : f16) outs(%54 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %58 = linalg.matmul ins(%41, %transposed : tensor<64x64xf16>, tensor<64x32xf16>) outs(%57 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %50 : memref<64x32xf16>
                  loom.semaphore_give %38 : memref<64x64xf16>
                  %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%58, %34 : tensor<64x32xf16>, tensor<64xf16>) outs(%56 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_9: f16, %out: f16):
                    %91 = arith.mulf %in_9, %cst_1 : f16
                    %92 = math.powf %cst, %91 : f16
                    %93 = arith.mulf %in, %92 : f16
                    linalg.yield %93 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %53 : memref<64x32xf16>
                  %60 = arith.addi %20, %c1 : index
                  %61 = arith.muli %60, %c64 : index
                  %62 = arith.ceildivui %61, %c64 : index
                  %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %65 = loom.init_tensor %64[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %66 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %67 = loom.init_tensor %66[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %68 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %69 = loom.semaphore_take %68 : memref<64x64xf16> -> memref<64x64xf16>
                  %70 = loom.init_tensor %69[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %71 = loom.alloc [64] on @L1 : memref<64xf16>
                  %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
                  %73 = loom.alloc [64] on @L1 : memref<64xf16>
                  %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
                  %75 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %76 = loom.semaphore_take %75 : memref<64x32xf16> -> memref<64x32xf16>
                  %77 = scf.for %arg14 = %c0 to %62 step %c1 iter_args(%arg15 = %59) -> (tensor<64x32xf16>) {
                    %91 = arith.muli %arg14, %c64 : index
                    %92 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %36, %23, %91)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%92], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%28, %31], LR : [%28, %33]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %93 = loom.bufferize_to_tensor %69[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %91)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%94], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%27, %31], LR : [%40, %33]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %95 = loom.bufferize_to_tensor %72[64] : memref<64xf16> -> tensor<64xf16>
                    %96 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %91)
                    %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%96], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_11, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%27, %31], LR : [%40, %33]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %97 = loom.bufferize_to_tensor %74[64] : memref<64xf16> -> tensor<64xf16>
                    %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%93, %34, %95, %97 : tensor<64x64xf16>, tensor<64xf16>, tensor<64xf16>, tensor<64xf16>) outs(%70 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_13: f16, %in_14: f16, %in_15: f16, %out: f16):
                      %104 = arith.mulf %in_14, %cst_1 : f16
                      %105 = arith.mulf %in_13, %cst_1 : f16
                      %106 = arith.subf %105, %104 : f16
                      %107 = math.powf %cst, %106 : f16
                      %108 = arith.mulf %in, %107 : f16
                      %109 = arith.mulf %108, %in_15 : f16
                      linalg.yield %109 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %74 : memref<64xf16>
                    loom.semaphore_give %72 : memref<64xf16>
                    %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %35, %21, %42)
                    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%99], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_12, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%27, %47], LR : [%40, %47]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %100 = loom.bufferize_to_tensor %76[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %101 = linalg.fill ins(%cst_0 : f16) outs(%67 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %102 = linalg.matmul ins(%98, %100 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%101 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %76 : memref<64x32xf16>
                    loom.semaphore_give %69 : memref<64x64xf16>
                    %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %102 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_13: f16, %out: f16):
                      %104 = arith.addf %in, %in_13 : f16
                      linalg.yield %104 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %66 : memref<64x32xf16>
                    scf.yield %103 : tensor<64x32xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %25 : memref<64xf16>
                  %78 = loom.alloc [1] on @L1 : memref<f16>
                  %79 = loom.semaphore_take %78 : memref<f16> -> memref<f16>
                  %80 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%80], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                  %81 = arith.addi %30, %c3 : index
                  loom.copy %reinterpret_cast_6, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %30], LR : [%c7, %81]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %82 = loom.bufferize_to_tensor %79[] : memref<f16> -> tensor<f16>
                  %83 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %35, %21, %42)
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %reinterpret_cast_7, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%27, %47], LR : [%40, %47]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                  %84 = loom.bufferize_to_tensor %64[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%77, %84, %82 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%65 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                    %91 = arith.mulf %in_9, %in_10 : f16
                    %92 = arith.addf %in, %91 : f16
                    linalg.yield %92 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %79 : memref<f16>
                  loom.semaphore_give %55 : memref<64x32xf16>
                  %86 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %35, %21, %42)
                  %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  %87 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %88 = loom.init_tensor %87[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %89 = loom.sync ins(%85 : tensor<64x32xf16>) outs(%88 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %90 = loom.bufferize_to_memref %89 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %90, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%27, %47], LR : [%40, %47]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.semaphore_give %87 : memref<64x32xf16>
                  loom.semaphore_give %64 : memref<64x32xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
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
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (4) {
                scf.for %arg13 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg13)
                  %21 = arith.muli %arg8, %c32 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %20, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg12, %c2 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg11, %c2 : index
                  %30 = arith.muli %arg8, %c4 : index
                  %31 = arith.addi %29, %30 : index
                  %32 = arith.addi %29, %c1 : index
                  %33 = arith.addi %32, %30 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%28, %31], LR : [%28, %33]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %34 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %35 = arith.muli %arg12, %c1024 : index
                  %36 = arith.divui %21, %c64 : index
                  %37 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %38 = loom.semaphore_take %37 : memref<64x64xf16> -> memref<64x64xf16>
                  %c0_2 = arith.constant 0 : index
                  %39 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %35, %36, %c0_2)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%39], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                  %40 = arith.addi %27, %c1 : index
                  loom.copy %reinterpret_cast_3, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%27, %31], LR : [%40, %33]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                  %41 = loom.bufferize_to_tensor %38[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %42 = arith.muli %arg10, %c32 : index
                  %43 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %44 = loom.semaphore_take %43 : memref<32x64xf16> -> memref<32x64xf16>
                  %c0_4 = arith.constant 0 : index
                  %45 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %22, %21, %42, %c0_4)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%45], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
                  %46 = arith.addi %arg10, %29 : index
                  %47 = arith.addi %46, %30 : index
                  loom.copy %reinterpret_cast_5, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%27, %47], LR : [%40, %47]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
                  %48 = loom.bufferize_to_tensor %44[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %49 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %50 = loom.semaphore_take %49 : memref<64x32xf16> -> memref<64x32xf16>
                  %51 = loom.init_tensor %50[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %transposed = linalg.transpose ins(%48 : tensor<32x64xf16>) outs(%51 : tensor<64x32xf16>) permutation = [1, 0] 
                  loom.semaphore_give %44 : memref<32x64xf16>
                  %52 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %53 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
                  %54 = loom.init_tensor %53[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %55 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
                  %56 = loom.init_tensor %55[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %57 = linalg.fill ins(%cst_0 : f16) outs(%54 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %58 = linalg.matmul ins(%41, %transposed : tensor<64x64xf16>, tensor<64x32xf16>) outs(%57 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %50 : memref<64x32xf16>
                  loom.semaphore_give %38 : memref<64x64xf16>
                  %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%58, %34 : tensor<64x32xf16>, tensor<64xf16>) outs(%56 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_9: f16, %out: f16):
                    %91 = arith.mulf %in_9, %cst_1 : f16
                    %92 = math.powf %cst, %91 : f16
                    %93 = arith.mulf %in, %92 : f16
                    linalg.yield %93 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %53 : memref<64x32xf16>
                  %60 = arith.addi %20, %c1 : index
                  %61 = arith.muli %60, %c64 : index
                  %62 = arith.ceildivui %61, %c64 : index
                  %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %65 = loom.init_tensor %64[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %66 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %67 = loom.init_tensor %66[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %68 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %69 = loom.semaphore_take %68 : memref<64x64xf16> -> memref<64x64xf16>
                  %70 = loom.init_tensor %69[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %71 = loom.alloc [64] on @L1 : memref<64xf16>
                  %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
                  %73 = loom.alloc [64] on @L1 : memref<64xf16>
                  %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
                  %75 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %76 = loom.semaphore_take %75 : memref<64x32xf16> -> memref<64x32xf16>
                  %77 = scf.for %arg14 = %c0 to %62 step %c1 iter_args(%arg15 = %59) -> (tensor<64x32xf16>) {
                    %91 = arith.muli %arg14, %c64 : index
                    %92 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %36, %23, %91)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%92], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%28, %31], LR : [%28, %33]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %93 = loom.bufferize_to_tensor %69[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %91)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%94], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%27, %31], LR : [%40, %33]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %95 = loom.bufferize_to_tensor %72[64] : memref<64xf16> -> tensor<64xf16>
                    %96 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %91)
                    %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%96], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_11, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%27, %31], LR : [%40, %33]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %97 = loom.bufferize_to_tensor %74[64] : memref<64xf16> -> tensor<64xf16>
                    %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%93, %34, %95, %97 : tensor<64x64xf16>, tensor<64xf16>, tensor<64xf16>, tensor<64xf16>) outs(%70 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_13: f16, %in_14: f16, %in_15: f16, %out: f16):
                      %104 = arith.mulf %in_14, %cst_1 : f16
                      %105 = arith.mulf %in_13, %cst_1 : f16
                      %106 = arith.subf %105, %104 : f16
                      %107 = math.powf %cst, %106 : f16
                      %108 = arith.mulf %in, %107 : f16
                      %109 = arith.mulf %108, %in_15 : f16
                      linalg.yield %109 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %74 : memref<64xf16>
                    loom.semaphore_give %72 : memref<64xf16>
                    %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %35, %21, %42)
                    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%99], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_12, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%27, %47], LR : [%40, %47]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %100 = loom.bufferize_to_tensor %76[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %101 = linalg.fill ins(%cst_0 : f16) outs(%67 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %102 = linalg.matmul ins(%98, %100 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%101 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %76 : memref<64x32xf16>
                    loom.semaphore_give %69 : memref<64x64xf16>
                    %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %102 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_13: f16, %out: f16):
                      %104 = arith.addf %in, %in_13 : f16
                      linalg.yield %104 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %66 : memref<64x32xf16>
                    scf.yield %103 : tensor<64x32xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %25 : memref<64xf16>
                  %78 = loom.alloc [1] on @L1 : memref<f16>
                  %79 = loom.semaphore_take %78 : memref<f16> -> memref<f16>
                  %80 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%80], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                  %81 = arith.addi %30, %c3 : index
                  loom.copy %reinterpret_cast_6, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %30], LR : [%c7, %81]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %82 = loom.bufferize_to_tensor %79[] : memref<f16> -> tensor<f16>
                  %83 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %35, %21, %42)
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %reinterpret_cast_7, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%27, %47], LR : [%40, %47]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                  %84 = loom.bufferize_to_tensor %64[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%77, %84, %82 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%65 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                    %91 = arith.mulf %in_9, %in_10 : f16
                    %92 = arith.addf %in, %91 : f16
                    linalg.yield %92 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %79 : memref<f16>
                  loom.semaphore_give %55 : memref<64x32xf16>
                  %86 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %35, %21, %42)
                  %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  %87 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %88 = loom.init_tensor %87[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %89 = loom.sync ins(%85 : tensor<64x32xf16>) outs(%88 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %90 = loom.bufferize_to_memref %89 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %90, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%27, %47], LR : [%40, %47]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.semaphore_give %87 : memref<64x32xf16>
                  loom.semaphore_give %64 : memref<64x32xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
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
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (4) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %20 = arith.muli %arg8, %c32 : index
                %21 = arith.muli %arg12, %c4 : index
                %22 = arith.muli %arg9, %c64 : index
                %23 = loom.alloc [64] on @L1 : memref<64xf16>
                %24 = loom.semaphore_take %23 : memref<64xf16> -> memref<64xf16>
                %25 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %20, %21, %22)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                %26 = arith.muli %arg11, %c4 : index
                %27 = arith.addi %arg9, %26 : index
                %28 = arith.muli %arg12, %c2 : index
                %29 = arith.muli %arg8, %c4 : index
                %30 = arith.addi %28, %29 : index
                %31 = arith.addi %28, %c1 : index
                %32 = arith.addi %31, %29 : index
                loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%27, %30], LR : [%27, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                %33 = loom.bufferize_to_tensor %24[64] : memref<64xf16> -> tensor<64xf16>
                %34 = arith.muli %arg12, %c1024 : index
                %35 = arith.divui %20, %c64 : index
                %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %c0_2 = arith.constant 0 : index
                %38 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %34, %35, %c0_2)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                %39 = arith.addi %26, %c3 : index
                loom.copy %reinterpret_cast_3, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%26, %30], LR : [%39, %32]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                %40 = loom.bufferize_to_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = arith.muli %arg10, %c32 : index
                %42 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                %43 = loom.semaphore_take %42 : memref<32x64xf16> -> memref<32x64xf16>
                %c0_4 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %21, %20, %41, %c0_4)
                %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
                %45 = arith.addi %arg10, %28 : index
                %46 = arith.addi %45, %29 : index
                loom.copy %reinterpret_cast_5, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%26, %46], LR : [%39, %46]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
                %47 = loom.bufferize_to_tensor %43[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                %48 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %49 = loom.semaphore_take %48 : memref<64x32xf16> -> memref<64x32xf16>
                %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %transposed = linalg.transpose ins(%47 : tensor<32x64xf16>) outs(%50 : tensor<64x32xf16>) permutation = [1, 0] 
                loom.semaphore_give %43 : memref<32x64xf16>
                %51 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %52 = loom.semaphore_take %51 : memref<64x32xf16> -> memref<64x32xf16>
                %53 = loom.init_tensor %52[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %54 = loom.semaphore_take %51 : memref<64x32xf16> -> memref<64x32xf16>
                %55 = loom.init_tensor %54[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %56 = linalg.fill ins(%cst_0 : f16) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                %57 = linalg.matmul ins(%40, %transposed : tensor<64x64xf16>, tensor<64x32xf16>) outs(%56 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %49 : memref<64x32xf16>
                loom.semaphore_give %37 : memref<64x64xf16>
                %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%57, %33 : tensor<64x32xf16>, tensor<64xf16>) outs(%55 : tensor<64x32xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %90 = arith.mulf %in_9, %cst_1 : f16
                  %91 = math.powf %cst, %90 : f16
                  %92 = arith.mulf %in, %91 : f16
                  linalg.yield %92 : f16
                } -> tensor<64x32xf16>
                loom.semaphore_give %52 : memref<64x32xf16>
                %59 = arith.addi %arg9, %c1 : index
                %60 = arith.muli %59, %c64 : index
                %61 = arith.ceildivui %60, %c64 : index
                %62 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %63 = loom.semaphore_take %62 : memref<64x32xf16> -> memref<64x32xf16>
                %64 = loom.init_tensor %63[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %65 = loom.semaphore_take %62 : memref<64x32xf16> -> memref<64x32xf16>
                %66 = loom.init_tensor %65[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %67 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %68 = loom.semaphore_take %67 : memref<64x64xf16> -> memref<64x64xf16>
                %69 = loom.init_tensor %68[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %70 = loom.alloc [64] on @L1 : memref<64xf16>
                %71 = loom.semaphore_take %70 : memref<64xf16> -> memref<64xf16>
                %72 = loom.alloc [64] on @L1 : memref<64xf16>
                %73 = loom.semaphore_take %72 : memref<64xf16> -> memref<64xf16>
                %74 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %75 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
                %76 = scf.for %arg13 = %c0 to %61 step %c1 iter_args(%arg14 = %58) -> (tensor<64x32xf16>) {
                  %90 = arith.muli %arg13, %c64 : index
                  %91 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %21, %35, %22, %90)
                  %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%91], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                  loom.copy %reinterpret_cast_9, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%27, %30], LR : [%27, %32]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                  %92 = loom.bufferize_to_tensor %68[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %93 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %20, %21, %90)
                  %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%93], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  loom.copy %reinterpret_cast_10, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%26, %30], LR : [%39, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %94 = loom.bufferize_to_tensor %71[64] : memref<64xf16> -> tensor<64xf16>
                  %95 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %20, %21, %90)
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%95], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  loom.copy %reinterpret_cast_11, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%26, %30], LR : [%39, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %96 = loom.bufferize_to_tensor %73[64] : memref<64xf16> -> tensor<64xf16>
                  %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%92, %33, %94, %96 : tensor<64x64xf16>, tensor<64xf16>, tensor<64xf16>, tensor<64xf16>) outs(%69 : tensor<64x64xf16>) {
                  ^bb0(%in: f16, %in_13: f16, %in_14: f16, %in_15: f16, %out: f16):
                    %103 = arith.mulf %in_14, %cst_1 : f16
                    %104 = arith.mulf %in_13, %cst_1 : f16
                    %105 = arith.subf %104, %103 : f16
                    %106 = math.powf %cst, %105 : f16
                    %107 = arith.mulf %in, %106 : f16
                    %108 = arith.mulf %107, %in_15 : f16
                    linalg.yield %108 : f16
                  } -> tensor<64x64xf16>
                  loom.semaphore_give %73 : memref<64xf16>
                  loom.semaphore_give %71 : memref<64xf16>
                  %98 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %20, %41)
                  %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%98], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %reinterpret_cast_12, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%26, %46], LR : [%39, %46]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                  %99 = loom.bufferize_to_tensor %75[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %100 = linalg.fill ins(%cst_0 : f16) outs(%66 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %101 = linalg.matmul ins(%97, %99 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%100 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %75 : memref<64x32xf16>
                  loom.semaphore_give %68 : memref<64x64xf16>
                  %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %101 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg14 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_13: f16, %out: f16):
                    %103 = arith.addf %in, %in_13 : f16
                    linalg.yield %103 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %65 : memref<64x32xf16>
                  scf.yield %102 : tensor<64x32xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %24 : memref<64xf16>
                %77 = loom.alloc [1] on @L1 : memref<f16>
                %78 = loom.semaphore_take %77 : memref<f16> -> memref<f16>
                %79 = affine.apply affine_map<(d0) -> (d0)>(%20)
                %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%79], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                %80 = arith.addi %29, %c3 : index
                loom.copy %reinterpret_cast_6, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %29], LR : [%c7, %80]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                %81 = loom.bufferize_to_tensor %78[] : memref<f16> -> tensor<f16>
                %82 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %20, %41)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%82], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%26, %46], LR : [%39, %46]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                %83 = loom.bufferize_to_tensor %63[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%76, %83, %81 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%64 : tensor<64x32xf16>) {
                ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                  %90 = arith.mulf %in_9, %in_10 : f16
                  %91 = arith.addf %in, %90 : f16
                  linalg.yield %91 : f16
                } -> tensor<64x32xf16>
                loom.semaphore_give %78 : memref<f16>
                loom.semaphore_give %54 : memref<64x32xf16>
                %85 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %20, %41)
                %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%85], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                %86 = loom.semaphore_take %62 : memref<64x32xf16> -> memref<64x32xf16>
                %87 = loom.init_tensor %86[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %88 = loom.sync ins(%84 : tensor<64x32xf16>) outs(%87 : tensor<64x32xf16>) -> tensor<64x32xf16>
                %89 = loom.bufferize_to_memref %88 : tensor<64x32xf16> -> memref<64x32xf16>
                loom.copy %89, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%26, %46], LR : [%39, %46]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %86 : memref<64x32xf16>
                loom.semaphore_give %63 : memref<64x32xf16>
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
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
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (4) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %20 = arith.muli %arg8, %c32 : index
                %21 = arith.muli %arg12, %c4 : index
                %22 = arith.muli %arg9, %c64 : index
                %23 = loom.alloc [64] on @L1 : memref<64xf16>
                %24 = loom.semaphore_take %23 : memref<64xf16> -> memref<64xf16>
                %25 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %20, %21, %22)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                %26 = arith.muli %arg12, %c4 : index
                %27 = arith.addi %arg9, %26 : index
                %28 = arith.muli %arg11, %c2 : index
                %29 = arith.muli %arg8, %c4 : index
                %30 = arith.addi %28, %29 : index
                %31 = arith.addi %28, %c1 : index
                %32 = arith.addi %31, %29 : index
                loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%27, %30], LR : [%27, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                %33 = loom.bufferize_to_tensor %24[64] : memref<64xf16> -> tensor<64xf16>
                %34 = arith.muli %arg12, %c1024 : index
                %35 = arith.divui %20, %c64 : index
                %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %c0_2 = arith.constant 0 : index
                %38 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %34, %35, %c0_2)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                %39 = arith.addi %26, %c3 : index
                loom.copy %reinterpret_cast_3, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%26, %30], LR : [%39, %32]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                %40 = loom.bufferize_to_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = arith.muli %arg10, %c32 : index
                %42 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                %43 = loom.semaphore_take %42 : memref<32x64xf16> -> memref<32x64xf16>
                %c0_4 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %21, %20, %41, %c0_4)
                %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
                %45 = arith.addi %arg10, %28 : index
                %46 = arith.addi %45, %29 : index
                loom.copy %reinterpret_cast_5, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%26, %46], LR : [%39, %46]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
                %47 = loom.bufferize_to_tensor %43[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                %48 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %49 = loom.semaphore_take %48 : memref<64x32xf16> -> memref<64x32xf16>
                %50 = loom.init_tensor %49[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %transposed = linalg.transpose ins(%47 : tensor<32x64xf16>) outs(%50 : tensor<64x32xf16>) permutation = [1, 0] 
                loom.semaphore_give %43 : memref<32x64xf16>
                %51 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %52 = loom.semaphore_take %51 : memref<64x32xf16> -> memref<64x32xf16>
                %53 = loom.init_tensor %52[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %54 = loom.semaphore_take %51 : memref<64x32xf16> -> memref<64x32xf16>
                %55 = loom.init_tensor %54[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %56 = linalg.fill ins(%cst_0 : f16) outs(%53 : tensor<64x32xf16>) -> tensor<64x32xf16>
                %57 = linalg.matmul ins(%40, %transposed : tensor<64x64xf16>, tensor<64x32xf16>) outs(%56 : tensor<64x32xf16>) -> tensor<64x32xf16>
                loom.semaphore_give %49 : memref<64x32xf16>
                loom.semaphore_give %37 : memref<64x64xf16>
                %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%57, %33 : tensor<64x32xf16>, tensor<64xf16>) outs(%55 : tensor<64x32xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %90 = arith.mulf %in_9, %cst_1 : f16
                  %91 = math.powf %cst, %90 : f16
                  %92 = arith.mulf %in, %91 : f16
                  linalg.yield %92 : f16
                } -> tensor<64x32xf16>
                loom.semaphore_give %52 : memref<64x32xf16>
                %59 = arith.addi %arg9, %c1 : index
                %60 = arith.muli %59, %c64 : index
                %61 = arith.ceildivui %60, %c64 : index
                %62 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %63 = loom.semaphore_take %62 : memref<64x32xf16> -> memref<64x32xf16>
                %64 = loom.init_tensor %63[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %65 = loom.semaphore_take %62 : memref<64x32xf16> -> memref<64x32xf16>
                %66 = loom.init_tensor %65[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %67 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %68 = loom.semaphore_take %67 : memref<64x64xf16> -> memref<64x64xf16>
                %69 = loom.init_tensor %68[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %70 = loom.alloc [64] on @L1 : memref<64xf16>
                %71 = loom.semaphore_take %70 : memref<64xf16> -> memref<64xf16>
                %72 = loom.alloc [64] on @L1 : memref<64xf16>
                %73 = loom.semaphore_take %72 : memref<64xf16> -> memref<64xf16>
                %74 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %75 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
                %76 = scf.for %arg13 = %c0 to %61 step %c1 iter_args(%arg14 = %58) -> (tensor<64x32xf16>) {
                  %90 = arith.muli %arg13, %c64 : index
                  %91 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %21, %35, %22, %90)
                  %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%91], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                  loom.copy %reinterpret_cast_9, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%27, %30], LR : [%27, %32]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                  %92 = loom.bufferize_to_tensor %68[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %93 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %20, %21, %90)
                  %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%93], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  loom.copy %reinterpret_cast_10, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%26, %30], LR : [%39, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %94 = loom.bufferize_to_tensor %71[64] : memref<64xf16> -> tensor<64xf16>
                  %95 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %20, %21, %90)
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%95], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  loom.copy %reinterpret_cast_11, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%26, %30], LR : [%39, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %96 = loom.bufferize_to_tensor %73[64] : memref<64xf16> -> tensor<64xf16>
                  %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%92, %33, %94, %96 : tensor<64x64xf16>, tensor<64xf16>, tensor<64xf16>, tensor<64xf16>) outs(%69 : tensor<64x64xf16>) {
                  ^bb0(%in: f16, %in_13: f16, %in_14: f16, %in_15: f16, %out: f16):
                    %103 = arith.mulf %in_14, %cst_1 : f16
                    %104 = arith.mulf %in_13, %cst_1 : f16
                    %105 = arith.subf %104, %103 : f16
                    %106 = math.powf %cst, %105 : f16
                    %107 = arith.mulf %in, %106 : f16
                    %108 = arith.mulf %107, %in_15 : f16
                    linalg.yield %108 : f16
                  } -> tensor<64x64xf16>
                  loom.semaphore_give %73 : memref<64xf16>
                  loom.semaphore_give %71 : memref<64xf16>
                  %98 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %20, %41)
                  %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%98], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %reinterpret_cast_12, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%26, %46], LR : [%39, %46]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                  %99 = loom.bufferize_to_tensor %75[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %100 = linalg.fill ins(%cst_0 : f16) outs(%66 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %101 = linalg.matmul ins(%97, %99 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%100 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %75 : memref<64x32xf16>
                  loom.semaphore_give %68 : memref<64x64xf16>
                  %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %101 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg14 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_13: f16, %out: f16):
                    %103 = arith.addf %in, %in_13 : f16
                    linalg.yield %103 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %65 : memref<64x32xf16>
                  scf.yield %102 : tensor<64x32xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %24 : memref<64xf16>
                %77 = loom.alloc [1] on @L1 : memref<f16>
                %78 = loom.semaphore_take %77 : memref<f16> -> memref<f16>
                %79 = affine.apply affine_map<(d0) -> (d0)>(%20)
                %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%79], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                %80 = arith.addi %29, %c3 : index
                loom.copy %reinterpret_cast_6, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %29], LR : [%c7, %80]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                %81 = loom.bufferize_to_tensor %78[] : memref<f16> -> tensor<f16>
                %82 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %20, %41)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%82], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%26, %46], LR : [%39, %46]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                %83 = loom.bufferize_to_tensor %63[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%76, %83, %81 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%64 : tensor<64x32xf16>) {
                ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                  %90 = arith.mulf %in_9, %in_10 : f16
                  %91 = arith.addf %in, %90 : f16
                  linalg.yield %91 : f16
                } -> tensor<64x32xf16>
                loom.semaphore_give %78 : memref<f16>
                loom.semaphore_give %54 : memref<64x32xf16>
                %85 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %20, %41)
                %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%85], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                %86 = loom.semaphore_take %62 : memref<64x32xf16> -> memref<64x32xf16>
                %87 = loom.init_tensor %86[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %88 = loom.sync ins(%84 : tensor<64x32xf16>) outs(%87 : tensor<64x32xf16>) -> tensor<64x32xf16>
                %89 = loom.bufferize_to_memref %88 : tensor<64x32xf16> -> memref<64x32xf16>
                loom.copy %89, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%26, %46], LR : [%39, %46]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %86 : memref<64x32xf16>
                loom.semaphore_give %63 : memref<64x32xf16>
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
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
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (4) {
                scf.for %arg13 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg13)
                  %21 = arith.muli %arg8, %c32 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %20, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg11, %c2 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg8, %c4 : index
                  %30 = arith.addi %28, %29 : index
                  %31 = arith.muli %arg12, %c2 : index
                  %32 = arith.addi %31, %c1 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%30, %31], LR : [%30, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %33 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %34 = arith.muli %arg12, %c1024 : index
                  %35 = arith.divui %21, %c64 : index
                  %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                  %c0_2 = arith.constant 0 : index
                  %38 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %34, %35, %c0_2)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                  %39 = arith.addi %27, %29 : index
                  %40 = arith.addi %27, %c1 : index
                  %41 = arith.addi %40, %29 : index
                  loom.copy %reinterpret_cast_3, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%39, %31], LR : [%41, %32]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                  %42 = loom.bufferize_to_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %43 = arith.muli %arg10, %c32 : index
                  %44 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %45 = loom.semaphore_take %44 : memref<32x64xf16> -> memref<32x64xf16>
                  %c0_4 = arith.constant 0 : index
                  %46 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %22, %21, %43, %c0_4)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%46], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
                  %47 = arith.addi %arg10, %31 : index
                  loom.copy %reinterpret_cast_5, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
                  %48 = loom.bufferize_to_tensor %45[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %49 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %50 = loom.semaphore_take %49 : memref<64x32xf16> -> memref<64x32xf16>
                  %51 = loom.init_tensor %50[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %transposed = linalg.transpose ins(%48 : tensor<32x64xf16>) outs(%51 : tensor<64x32xf16>) permutation = [1, 0] 
                  loom.semaphore_give %45 : memref<32x64xf16>
                  %52 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %53 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
                  %54 = loom.init_tensor %53[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %55 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
                  %56 = loom.init_tensor %55[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %57 = linalg.fill ins(%cst_0 : f16) outs(%54 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %58 = linalg.matmul ins(%42, %transposed : tensor<64x64xf16>, tensor<64x32xf16>) outs(%57 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %50 : memref<64x32xf16>
                  loom.semaphore_give %37 : memref<64x64xf16>
                  %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%58, %33 : tensor<64x32xf16>, tensor<64xf16>) outs(%56 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_9: f16, %out: f16):
                    %91 = arith.mulf %in_9, %cst_1 : f16
                    %92 = math.powf %cst, %91 : f16
                    %93 = arith.mulf %in, %92 : f16
                    linalg.yield %93 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %53 : memref<64x32xf16>
                  %60 = arith.addi %20, %c1 : index
                  %61 = arith.muli %60, %c64 : index
                  %62 = arith.ceildivui %61, %c64 : index
                  %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %65 = loom.init_tensor %64[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %66 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %67 = loom.init_tensor %66[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %68 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %69 = loom.semaphore_take %68 : memref<64x64xf16> -> memref<64x64xf16>
                  %70 = loom.init_tensor %69[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %71 = loom.alloc [64] on @L1 : memref<64xf16>
                  %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
                  %73 = loom.alloc [64] on @L1 : memref<64xf16>
                  %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
                  %75 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %76 = loom.semaphore_take %75 : memref<64x32xf16> -> memref<64x32xf16>
                  %77 = scf.for %arg14 = %c0 to %62 step %c1 iter_args(%arg15 = %59) -> (tensor<64x32xf16>) {
                    %91 = arith.muli %arg14, %c64 : index
                    %92 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %35, %23, %91)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%92], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%30, %31], LR : [%30, %32]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %93 = loom.bufferize_to_tensor %69[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %91)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%94], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%39, %31], LR : [%41, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %95 = loom.bufferize_to_tensor %72[64] : memref<64xf16> -> tensor<64xf16>
                    %96 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %91)
                    %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%96], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_11, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%39, %31], LR : [%41, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %97 = loom.bufferize_to_tensor %74[64] : memref<64xf16> -> tensor<64xf16>
                    %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%93, %33, %95, %97 : tensor<64x64xf16>, tensor<64xf16>, tensor<64xf16>, tensor<64xf16>) outs(%70 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_13: f16, %in_14: f16, %in_15: f16, %out: f16):
                      %104 = arith.mulf %in_14, %cst_1 : f16
                      %105 = arith.mulf %in_13, %cst_1 : f16
                      %106 = arith.subf %105, %104 : f16
                      %107 = math.powf %cst, %106 : f16
                      %108 = arith.mulf %in, %107 : f16
                      %109 = arith.mulf %108, %in_15 : f16
                      linalg.yield %109 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %74 : memref<64xf16>
                    loom.semaphore_give %72 : memref<64xf16>
                    %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %21, %43)
                    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%99], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_12, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %100 = loom.bufferize_to_tensor %76[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %101 = linalg.fill ins(%cst_0 : f16) outs(%67 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %102 = linalg.matmul ins(%98, %100 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%101 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %76 : memref<64x32xf16>
                    loom.semaphore_give %69 : memref<64x64xf16>
                    %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %102 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_13: f16, %out: f16):
                      %104 = arith.addf %in, %in_13 : f16
                      linalg.yield %104 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %66 : memref<64x32xf16>
                    scf.yield %103 : tensor<64x32xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %25 : memref<64xf16>
                  %78 = loom.alloc [1] on @L1 : memref<f16>
                  %79 = loom.semaphore_take %78 : memref<f16> -> memref<f16>
                  %80 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%80], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                  %81 = arith.addi %29, %c3 : index
                  loom.copy %reinterpret_cast_6, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%29, %c0], LR : [%81, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %82 = loom.bufferize_to_tensor %79[] : memref<f16> -> tensor<f16>
                  %83 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %21, %43)
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %reinterpret_cast_7, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                  %84 = loom.bufferize_to_tensor %64[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%77, %84, %82 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%65 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                    %91 = arith.mulf %in_9, %in_10 : f16
                    %92 = arith.addf %in, %91 : f16
                    linalg.yield %92 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %79 : memref<f16>
                  loom.semaphore_give %55 : memref<64x32xf16>
                  %86 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %21, %43)
                  %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  %87 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %88 = loom.init_tensor %87[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %89 = loom.sync ins(%85 : tensor<64x32xf16>) outs(%88 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %90 = loom.bufferize_to_memref %89 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %90, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.semaphore_give %87 : memref<64x32xf16>
                  loom.semaphore_give %64 : memref<64x32xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
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
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (4) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg13)
                  %21 = arith.muli %arg8, %c32 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %20, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg12, %c2 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg8, %c4 : index
                  %30 = arith.addi %28, %29 : index
                  %31 = arith.muli %arg11, %c2 : index
                  %32 = arith.addi %31, %c1 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%30, %31], LR : [%30, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %33 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %34 = arith.muli %arg12, %c1024 : index
                  %35 = arith.divui %21, %c64 : index
                  %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                  %c0_2 = arith.constant 0 : index
                  %38 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %34, %35, %c0_2)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                  %39 = arith.addi %27, %29 : index
                  %40 = arith.addi %27, %c1 : index
                  %41 = arith.addi %40, %29 : index
                  loom.copy %reinterpret_cast_3, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%39, %31], LR : [%41, %32]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                  %42 = loom.bufferize_to_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %43 = arith.muli %arg10, %c32 : index
                  %44 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %45 = loom.semaphore_take %44 : memref<32x64xf16> -> memref<32x64xf16>
                  %c0_4 = arith.constant 0 : index
                  %46 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %22, %21, %43, %c0_4)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%46], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
                  %47 = arith.addi %arg10, %31 : index
                  loom.copy %reinterpret_cast_5, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
                  %48 = loom.bufferize_to_tensor %45[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %49 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %50 = loom.semaphore_take %49 : memref<64x32xf16> -> memref<64x32xf16>
                  %51 = loom.init_tensor %50[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %transposed = linalg.transpose ins(%48 : tensor<32x64xf16>) outs(%51 : tensor<64x32xf16>) permutation = [1, 0] 
                  loom.semaphore_give %45 : memref<32x64xf16>
                  %52 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %53 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
                  %54 = loom.init_tensor %53[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %55 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
                  %56 = loom.init_tensor %55[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %57 = linalg.fill ins(%cst_0 : f16) outs(%54 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %58 = linalg.matmul ins(%42, %transposed : tensor<64x64xf16>, tensor<64x32xf16>) outs(%57 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %50 : memref<64x32xf16>
                  loom.semaphore_give %37 : memref<64x64xf16>
                  %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%58, %33 : tensor<64x32xf16>, tensor<64xf16>) outs(%56 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_9: f16, %out: f16):
                    %91 = arith.mulf %in_9, %cst_1 : f16
                    %92 = math.powf %cst, %91 : f16
                    %93 = arith.mulf %in, %92 : f16
                    linalg.yield %93 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %53 : memref<64x32xf16>
                  %60 = arith.addi %20, %c1 : index
                  %61 = arith.muli %60, %c64 : index
                  %62 = arith.ceildivui %61, %c64 : index
                  %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %65 = loom.init_tensor %64[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %66 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %67 = loom.init_tensor %66[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %68 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %69 = loom.semaphore_take %68 : memref<64x64xf16> -> memref<64x64xf16>
                  %70 = loom.init_tensor %69[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %71 = loom.alloc [64] on @L1 : memref<64xf16>
                  %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
                  %73 = loom.alloc [64] on @L1 : memref<64xf16>
                  %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
                  %75 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %76 = loom.semaphore_take %75 : memref<64x32xf16> -> memref<64x32xf16>
                  %77 = scf.for %arg14 = %c0 to %62 step %c1 iter_args(%arg15 = %59) -> (tensor<64x32xf16>) {
                    %91 = arith.muli %arg14, %c64 : index
                    %92 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %35, %23, %91)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%92], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%30, %31], LR : [%30, %32]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %93 = loom.bufferize_to_tensor %69[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %91)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%94], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%39, %31], LR : [%41, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %95 = loom.bufferize_to_tensor %72[64] : memref<64xf16> -> tensor<64xf16>
                    %96 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %91)
                    %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%96], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_11, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%39, %31], LR : [%41, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %97 = loom.bufferize_to_tensor %74[64] : memref<64xf16> -> tensor<64xf16>
                    %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%93, %33, %95, %97 : tensor<64x64xf16>, tensor<64xf16>, tensor<64xf16>, tensor<64xf16>) outs(%70 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_13: f16, %in_14: f16, %in_15: f16, %out: f16):
                      %104 = arith.mulf %in_14, %cst_1 : f16
                      %105 = arith.mulf %in_13, %cst_1 : f16
                      %106 = arith.subf %105, %104 : f16
                      %107 = math.powf %cst, %106 : f16
                      %108 = arith.mulf %in, %107 : f16
                      %109 = arith.mulf %108, %in_15 : f16
                      linalg.yield %109 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %74 : memref<64xf16>
                    loom.semaphore_give %72 : memref<64xf16>
                    %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %21, %43)
                    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%99], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_12, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %100 = loom.bufferize_to_tensor %76[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %101 = linalg.fill ins(%cst_0 : f16) outs(%67 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %102 = linalg.matmul ins(%98, %100 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%101 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %76 : memref<64x32xf16>
                    loom.semaphore_give %69 : memref<64x64xf16>
                    %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %102 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_13: f16, %out: f16):
                      %104 = arith.addf %in, %in_13 : f16
                      linalg.yield %104 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %66 : memref<64x32xf16>
                    scf.yield %103 : tensor<64x32xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %25 : memref<64xf16>
                  %78 = loom.alloc [1] on @L1 : memref<f16>
                  %79 = loom.semaphore_take %78 : memref<f16> -> memref<f16>
                  %80 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%80], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                  %81 = arith.addi %29, %c3 : index
                  loom.copy %reinterpret_cast_6, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%29, %c0], LR : [%81, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %82 = loom.bufferize_to_tensor %79[] : memref<f16> -> tensor<f16>
                  %83 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %21, %43)
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %reinterpret_cast_7, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                  %84 = loom.bufferize_to_tensor %64[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%77, %84, %82 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%65 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                    %91 = arith.mulf %in_9, %in_10 : f16
                    %92 = arith.addf %in, %91 : f16
                    linalg.yield %92 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %79 : memref<f16>
                  loom.semaphore_give %55 : memref<64x32xf16>
                  %86 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %21, %43)
                  %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  %87 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %88 = loom.init_tensor %87[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %89 = loom.sync ins(%85 : tensor<64x32xf16>) outs(%88 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %90 = loom.bufferize_to_memref %89 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %90, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.semaphore_give %87 : memref<64x32xf16>
                  loom.semaphore_give %64 : memref<64x32xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
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
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (4) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg13)
                  %21 = arith.muli %arg8, %c32 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %20, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg11, %c2 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg8, %c4 : index
                  %30 = arith.addi %28, %29 : index
                  %31 = arith.muli %arg12, %c4 : index
                  %32 = arith.addi %31, %c3 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%30, %31], LR : [%30, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %33 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %34 = arith.muli %arg12, %c1024 : index
                  %35 = arith.divui %21, %c64 : index
                  %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                  %c0_2 = arith.constant 0 : index
                  %38 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %34, %35, %c0_2)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                  %39 = arith.addi %27, %29 : index
                  %40 = arith.addi %27, %c1 : index
                  %41 = arith.addi %40, %29 : index
                  loom.copy %reinterpret_cast_3, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%39, %31], LR : [%41, %32]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                  %42 = loom.bufferize_to_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %43 = arith.muli %arg10, %c32 : index
                  %44 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %45 = loom.semaphore_take %44 : memref<32x64xf16> -> memref<32x64xf16>
                  %c0_4 = arith.constant 0 : index
                  %46 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %22, %21, %43, %c0_4)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%46], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
                  %47 = arith.addi %arg10, %31 : index
                  loom.copy %reinterpret_cast_5, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
                  %48 = loom.bufferize_to_tensor %45[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %49 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %50 = loom.semaphore_take %49 : memref<64x32xf16> -> memref<64x32xf16>
                  %51 = loom.init_tensor %50[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %transposed = linalg.transpose ins(%48 : tensor<32x64xf16>) outs(%51 : tensor<64x32xf16>) permutation = [1, 0] 
                  loom.semaphore_give %45 : memref<32x64xf16>
                  %52 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %53 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
                  %54 = loom.init_tensor %53[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %55 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
                  %56 = loom.init_tensor %55[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %57 = linalg.fill ins(%cst_0 : f16) outs(%54 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %58 = linalg.matmul ins(%42, %transposed : tensor<64x64xf16>, tensor<64x32xf16>) outs(%57 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %50 : memref<64x32xf16>
                  loom.semaphore_give %37 : memref<64x64xf16>
                  %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%58, %33 : tensor<64x32xf16>, tensor<64xf16>) outs(%56 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_9: f16, %out: f16):
                    %91 = arith.mulf %in_9, %cst_1 : f16
                    %92 = math.powf %cst, %91 : f16
                    %93 = arith.mulf %in, %92 : f16
                    linalg.yield %93 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %53 : memref<64x32xf16>
                  %60 = arith.addi %20, %c1 : index
                  %61 = arith.muli %60, %c64 : index
                  %62 = arith.ceildivui %61, %c64 : index
                  %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %65 = loom.init_tensor %64[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %66 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %67 = loom.init_tensor %66[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %68 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %69 = loom.semaphore_take %68 : memref<64x64xf16> -> memref<64x64xf16>
                  %70 = loom.init_tensor %69[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %71 = loom.alloc [64] on @L1 : memref<64xf16>
                  %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
                  %73 = loom.alloc [64] on @L1 : memref<64xf16>
                  %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
                  %75 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %76 = loom.semaphore_take %75 : memref<64x32xf16> -> memref<64x32xf16>
                  %77 = scf.for %arg14 = %c0 to %62 step %c1 iter_args(%arg15 = %59) -> (tensor<64x32xf16>) {
                    %91 = arith.muli %arg14, %c64 : index
                    %92 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %35, %23, %91)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%92], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%30, %31], LR : [%30, %32]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %93 = loom.bufferize_to_tensor %69[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %91)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%94], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%39, %31], LR : [%41, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %95 = loom.bufferize_to_tensor %72[64] : memref<64xf16> -> tensor<64xf16>
                    %96 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %91)
                    %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%96], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_11, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%39, %31], LR : [%41, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %97 = loom.bufferize_to_tensor %74[64] : memref<64xf16> -> tensor<64xf16>
                    %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%93, %33, %95, %97 : tensor<64x64xf16>, tensor<64xf16>, tensor<64xf16>, tensor<64xf16>) outs(%70 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_13: f16, %in_14: f16, %in_15: f16, %out: f16):
                      %104 = arith.mulf %in_14, %cst_1 : f16
                      %105 = arith.mulf %in_13, %cst_1 : f16
                      %106 = arith.subf %105, %104 : f16
                      %107 = math.powf %cst, %106 : f16
                      %108 = arith.mulf %in, %107 : f16
                      %109 = arith.mulf %108, %in_15 : f16
                      linalg.yield %109 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %74 : memref<64xf16>
                    loom.semaphore_give %72 : memref<64xf16>
                    %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %21, %43)
                    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%99], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_12, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %100 = loom.bufferize_to_tensor %76[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %101 = linalg.fill ins(%cst_0 : f16) outs(%67 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %102 = linalg.matmul ins(%98, %100 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%101 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %76 : memref<64x32xf16>
                    loom.semaphore_give %69 : memref<64x64xf16>
                    %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %102 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_13: f16, %out: f16):
                      %104 = arith.addf %in, %in_13 : f16
                      linalg.yield %104 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %66 : memref<64x32xf16>
                    scf.yield %103 : tensor<64x32xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %25 : memref<64xf16>
                  %78 = loom.alloc [1] on @L1 : memref<f16>
                  %79 = loom.semaphore_take %78 : memref<f16> -> memref<f16>
                  %80 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%80], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                  %81 = arith.addi %29, %c3 : index
                  loom.copy %reinterpret_cast_6, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%29, %c0], LR : [%81, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %82 = loom.bufferize_to_tensor %79[] : memref<f16> -> tensor<f16>
                  %83 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %21, %43)
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %reinterpret_cast_7, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                  %84 = loom.bufferize_to_tensor %64[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%77, %84, %82 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%65 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                    %91 = arith.mulf %in_9, %in_10 : f16
                    %92 = arith.addf %in, %91 : f16
                    linalg.yield %92 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %79 : memref<f16>
                  loom.semaphore_give %55 : memref<64x32xf16>
                  %86 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %21, %43)
                  %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  %87 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %88 = loom.init_tensor %87[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %89 = loom.sync ins(%85 : tensor<64x32xf16>) outs(%88 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %90 = loom.bufferize_to_memref %89 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %90, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.semaphore_give %87 : memref<64x32xf16>
                  loom.semaphore_give %64 : memref<64x32xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h32__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
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
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (4) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg13)
                  %21 = arith.muli %arg8, %c32 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %20, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg12, %c2 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg8, %c4 : index
                  %30 = arith.addi %28, %29 : index
                  %31 = arith.muli %arg11, %c4 : index
                  %32 = arith.addi %31, %c3 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%30, %31], LR : [%30, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %33 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %34 = arith.muli %arg12, %c1024 : index
                  %35 = arith.divui %21, %c64 : index
                  %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                  %c0_2 = arith.constant 0 : index
                  %38 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %34, %35, %c0_2)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                  %39 = arith.addi %27, %29 : index
                  %40 = arith.addi %27, %c1 : index
                  %41 = arith.addi %40, %29 : index
                  loom.copy %reinterpret_cast_3, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%39, %31], LR : [%41, %32]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                  %42 = loom.bufferize_to_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %43 = arith.muli %arg10, %c32 : index
                  %44 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %45 = loom.semaphore_take %44 : memref<32x64xf16> -> memref<32x64xf16>
                  %c0_4 = arith.constant 0 : index
                  %46 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %22, %21, %43, %c0_4)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%46], sizes: [32, 64], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
                  %47 = arith.addi %arg10, %31 : index
                  loom.copy %reinterpret_cast_5, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
                  %48 = loom.bufferize_to_tensor %45[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %49 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %50 = loom.semaphore_take %49 : memref<64x32xf16> -> memref<64x32xf16>
                  %51 = loom.init_tensor %50[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %transposed = linalg.transpose ins(%48 : tensor<32x64xf16>) outs(%51 : tensor<64x32xf16>) permutation = [1, 0] 
                  loom.semaphore_give %45 : memref<32x64xf16>
                  %52 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %53 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
                  %54 = loom.init_tensor %53[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %55 = loom.semaphore_take %52 : memref<64x32xf16> -> memref<64x32xf16>
                  %56 = loom.init_tensor %55[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %57 = linalg.fill ins(%cst_0 : f16) outs(%54 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %58 = linalg.matmul ins(%42, %transposed : tensor<64x64xf16>, tensor<64x32xf16>) outs(%57 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %50 : memref<64x32xf16>
                  loom.semaphore_give %37 : memref<64x64xf16>
                  %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%58, %33 : tensor<64x32xf16>, tensor<64xf16>) outs(%56 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_9: f16, %out: f16):
                    %91 = arith.mulf %in_9, %cst_1 : f16
                    %92 = math.powf %cst, %91 : f16
                    %93 = arith.mulf %in, %92 : f16
                    linalg.yield %93 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %53 : memref<64x32xf16>
                  %60 = arith.addi %20, %c1 : index
                  %61 = arith.muli %60, %c64 : index
                  %62 = arith.ceildivui %61, %c64 : index
                  %63 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %64 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %65 = loom.init_tensor %64[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %66 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %67 = loom.init_tensor %66[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %68 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %69 = loom.semaphore_take %68 : memref<64x64xf16> -> memref<64x64xf16>
                  %70 = loom.init_tensor %69[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %71 = loom.alloc [64] on @L1 : memref<64xf16>
                  %72 = loom.semaphore_take %71 : memref<64xf16> -> memref<64xf16>
                  %73 = loom.alloc [64] on @L1 : memref<64xf16>
                  %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
                  %75 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %76 = loom.semaphore_take %75 : memref<64x32xf16> -> memref<64x32xf16>
                  %77 = scf.for %arg14 = %c0 to %62 step %c1 iter_args(%arg15 = %59) -> (tensor<64x32xf16>) {
                    %91 = arith.muli %arg14, %c64 : index
                    %92 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %35, %23, %91)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%92], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%30, %31], LR : [%30, %32]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %93 = loom.bufferize_to_tensor %69[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %91)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%94], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%39, %31], LR : [%41, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %95 = loom.bufferize_to_tensor %72[64] : memref<64xf16> -> tensor<64xf16>
                    %96 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %91)
                    %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%96], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_11, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%39, %31], LR : [%41, %32]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %97 = loom.bufferize_to_tensor %74[64] : memref<64xf16> -> tensor<64xf16>
                    %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%93, %33, %95, %97 : tensor<64x64xf16>, tensor<64xf16>, tensor<64xf16>, tensor<64xf16>) outs(%70 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_13: f16, %in_14: f16, %in_15: f16, %out: f16):
                      %104 = arith.mulf %in_14, %cst_1 : f16
                      %105 = arith.mulf %in_13, %cst_1 : f16
                      %106 = arith.subf %105, %104 : f16
                      %107 = math.powf %cst, %106 : f16
                      %108 = arith.mulf %in, %107 : f16
                      %109 = arith.mulf %108, %in_15 : f16
                      linalg.yield %109 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %74 : memref<64xf16>
                    loom.semaphore_give %72 : memref<64xf16>
                    %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %21, %43)
                    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%99], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_12, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %100 = loom.bufferize_to_tensor %76[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %101 = linalg.fill ins(%cst_0 : f16) outs(%67 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %102 = linalg.matmul ins(%98, %100 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%101 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %76 : memref<64x32xf16>
                    loom.semaphore_give %69 : memref<64x64xf16>
                    %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %102 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_13: f16, %out: f16):
                      %104 = arith.addf %in, %in_13 : f16
                      linalg.yield %104 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %66 : memref<64x32xf16>
                    scf.yield %103 : tensor<64x32xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %25 : memref<64xf16>
                  %78 = loom.alloc [1] on @L1 : memref<f16>
                  %79 = loom.semaphore_take %78 : memref<f16> -> memref<f16>
                  %80 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%80], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                  %81 = arith.addi %29, %c3 : index
                  loom.copy %reinterpret_cast_6, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%29, %c0], LR : [%81, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %82 = loom.bufferize_to_tensor %79[] : memref<f16> -> tensor<f16>
                  %83 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %21, %43)
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %reinterpret_cast_7, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                  %84 = loom.bufferize_to_tensor %64[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%77, %84, %82 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%65 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                    %91 = arith.mulf %in_9, %in_10 : f16
                    %92 = arith.addf %in, %91 : f16
                    linalg.yield %92 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %79 : memref<f16>
                  loom.semaphore_give %55 : memref<64x32xf16>
                  %86 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %34, %21, %43)
                  %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%86], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  %87 = loom.semaphore_take %63 : memref<64x32xf16> -> memref<64x32xf16>
                  %88 = loom.init_tensor %87[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %89 = loom.sync ins(%85 : tensor<64x32xf16>) outs(%88 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %90 = loom.bufferize_to_memref %89 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %90, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%39, %47], LR : [%41, %47]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.semaphore_give %87 : memref<64x32xf16>
                  loom.semaphore_give %64 : memref<64x32xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
