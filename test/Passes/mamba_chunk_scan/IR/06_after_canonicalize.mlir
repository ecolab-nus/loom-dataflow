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
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (4) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c4 step %c1 {
                  scf.for %arg14 = %c0 to %c2 step %c1 {
                    %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                    %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                    %22 = arith.muli %20, %c8 : index
                    %23 = arith.muli %arg12, %c4 : index
                    %24 = arith.muli %21, %c64 : index
                    %25 = loom.alloc [64] on @L1 : memref<64xf16>
                    %26 = loom.semaphore_take %25 : memref<64xf16> -> memref<64xf16>
                    %27 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    %28 = arith.muli %arg11, %c2 : index
                    %29 = arith.addi %arg9, %28 : index
                    %30 = arith.muli %arg12, %c2 : index
                    %31 = arith.muli %arg8, %c4 : index
                    %32 = arith.addi %30, %31 : index
                    %33 = arith.addi %30, %c1 : index
                    %34 = arith.addi %33, %31 : index
                    loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%29, %32], LR : [%29, %34]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %35 = loom.bufferize_to_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                    %36 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %37 = loom.semaphore_take %36 : memref<64x32xf16> -> memref<64x32xf16>
                    %38 = loom.init_tensor %37[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %39 = loom.semaphore_take %36 : memref<64x32xf16> -> memref<64x32xf16>
                    %40 = loom.init_tensor %39[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %41 = loom.broadcast ins(%35 : tensor<64xf16>) outs(%40 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                    %42 = arith.muli %arg12, %c1024 : index
                    %43 = arith.addi %24, %42 : index
                    %44 = arith.divui %22, %c64 : index
                    %45 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %46 = loom.semaphore_take %45 : memref<64x64xf16> -> memref<64x64xf16>
                    %c0_0 = arith.constant 0 : index
                    %47 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %43, %44, %c0_0)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                    loom.copy %reinterpret_cast_1, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%29, %32], LR : [%29, %34]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                    %48 = loom.bufferize_to_tensor %46[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %49 = arith.muli %arg10, %c32 : index
                    %50 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %51 = loom.semaphore_take %50 : memref<64x32xf16> -> memref<64x32xf16>
                    %c0_2 = arith.constant 0 : index
                    %52 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %23, %22, %c0_2, %49)
                    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%52], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                    %53 = arith.addi %28, %c1 : index
                    %54 = arith.addi %arg10, %30 : index
                    %55 = arith.addi %54, %31 : index
                    loom.copy %reinterpret_cast_3, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%28, %55], LR : [%53, %55]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                    %56 = loom.bufferize_to_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %57 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %58 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %59 = loom.init_tensor %58[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %60 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %61 = loom.init_tensor %60[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %62 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %63 = loom.init_tensor %62[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %64 = linalg.fill ins(%cst : f16) outs(%61 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %65 = linalg.matmul ins(%48, %56 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%64 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %51 : memref<64x32xf16>
                    loom.semaphore_give %46 : memref<64x64xf16>
                    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %41 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%63 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %out: f16):
                      %101 = math.exp %in_7 : f16
                      %102 = arith.mulf %in, %101 : f16
                      linalg.yield %102 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %60 : memref<64x32xf16>
                    loom.semaphore_give %39 : memref<64x32xf16>
                    %67 = arith.addi %21, %c1 : index
                    %68 = arith.muli %67, %c64 : index
                    %69 = arith.ceildivui %68, %c64 : index
                    %70 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %71 = loom.semaphore_take %70 : memref<64x32xf16> -> memref<64x32xf16>
                    %72 = loom.init_tensor %71[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %73 = loom.semaphore_take %70 : memref<64x32xf16> -> memref<64x32xf16>
                    %74 = loom.init_tensor %73[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %75 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %76 = loom.semaphore_take %75 : memref<64x64xf16> -> memref<64x64xf16>
                    %77 = loom.init_tensor %76[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %78 = loom.alloc [64] on @L1 : memref<64xf16>
                    %79 = loom.semaphore_take %78 : memref<64xf16> -> memref<64xf16>
                    %80 = loom.semaphore_take %78 : memref<64xf16> -> memref<64xf16>
                    %81 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %82 = loom.semaphore_take %81 : memref<32x64xf16> -> memref<32x64xf16>
                    %83 = loom.init_tensor %82[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %84 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %85 = loom.semaphore_take %84 : memref<32x64xf16> -> memref<32x64xf16>
                    %86 = loom.init_tensor %85[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %87 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %88 = loom.semaphore_take %87 : memref<64x32xf16> -> memref<64x32xf16>
                    %89 = scf.for %arg15 = %c0 to %69 step %c1 iter_args(%arg16 = %66) -> (tensor<64x32xf16>) {
                      %101 = arith.muli %arg15, %c64 : index
                      %102 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %23, %44, %24, %101)
                      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                      loom.copy %reinterpret_cast_7, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%29, %55], LR : [%29, %55]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                      %103 = loom.bufferize_to_tensor %76[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                      %104 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %101)
                      %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%104], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_8, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%29, %55], LR : [%29, %55]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %105 = loom.bufferize_to_tensor %80[64] : memref<64xf16> -> tensor<64xf16>
                      %106 = loom.broadcast ins(%35 : tensor<64xf16>) outs(%38 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                      %107 = loom.broadcast ins(%105 : tensor<64xf16>) outs(%83 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %80 : memref<64xf16>
                      %108 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %101)
                      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%108], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_9, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%29, %55], LR : [%29, %55]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %109 = loom.bufferize_to_tensor %79[64] : memref<64xf16> -> tensor<64xf16>
                      %110 = loom.broadcast ins(%109 : tensor<64xf16>) outs(%86 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %79 : memref<64xf16>
                      %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%103, %106, %107, %110 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%77 : tensor<64x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                        %118 = arith.subf %in_11, %in_12 : f16
                        %119 = math.exp %118 : f16
                        %120 = arith.mulf %in, %119 : f16
                        %121 = arith.mulf %120, %in_13 : f16
                        linalg.yield %121 : f16
                      } -> tensor<64x64xf16>
                      loom.semaphore_give %85 : memref<32x64xf16>
                      loom.semaphore_give %82 : memref<32x64xf16>
                      loom.semaphore_give %37 : memref<64x32xf16>
                      %112 = arith.addi %101, %42 : index
                      %113 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %112, %22, %49)
                      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                      loom.copy %reinterpret_cast_10, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%29, %55], LR : [%29, %55]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                      %114 = loom.bufferize_to_tensor %88[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                      %115 = linalg.fill ins(%cst : f16) outs(%74 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %116 = linalg.matmul ins(%111, %114 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%115 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %88 : memref<64x32xf16>
                      loom.semaphore_give %76 : memref<64x64xf16>
                      %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %116 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg16 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %118 = arith.addf %in, %in_11 : f16
                        linalg.yield %118 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %73 : memref<64x32xf16>
                      scf.yield %117 : tensor<64x32xf16>
                    } {loom.iter_type = #loom.iter_type<sequential>}
                    loom.semaphore_give %26 : memref<64xf16>
                    %90 = loom.alloc [1] on @L1 : memref<f16>
                    %91 = loom.semaphore_take %90 : memref<f16> -> memref<f16>
                    %92 = affine.apply affine_map<(d0) -> (d0)>(%22)
                    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%92], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                    %93 = arith.addi %31, %c3 : index
                    loom.copy %reinterpret_cast_4, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %31], LR : [%c7, %93]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                    %94 = loom.bufferize_to_tensor %91[] : memref<f16> -> tensor<f16>
                    %95 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %43, %22, %49)
                    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%95], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_5, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%29, %55], LR : [%29, %55]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %96 = loom.bufferize_to_tensor %71[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%89, %96, %94 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%72 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                      %101 = arith.mulf %in_7, %in_8 : f16
                      %102 = arith.addf %in, %101 : f16
                      linalg.yield %102 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %91 : memref<f16>
                    loom.semaphore_give %62 : memref<64x32xf16>
                    %98 = loom.sync ins(%97 : tensor<64x32xf16>) outs(%59 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %71 : memref<64x32xf16>
                    %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %43, %22, %49)
                    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%99], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    %100 = loom.bufferize_to_memref %98 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %100, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%29, %55], LR : [%29, %55]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.semaphore_give %58 : memref<64x32xf16>
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (4) {
                scf.for %arg13 = %c0 to %c4 step %c1 {
                  scf.for %arg14 = %c0 to %c2 step %c1 {
                    %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                    %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                    %22 = arith.muli %20, %c8 : index
                    %23 = arith.muli %arg12, %c4 : index
                    %24 = arith.muli %21, %c64 : index
                    %25 = loom.alloc [64] on @L1 : memref<64xf16>
                    %26 = loom.semaphore_take %25 : memref<64xf16> -> memref<64xf16>
                    %27 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    %28 = arith.muli %arg12, %c2 : index
                    %29 = arith.addi %arg9, %28 : index
                    %30 = arith.muli %arg11, %c2 : index
                    %31 = arith.muli %arg8, %c4 : index
                    %32 = arith.addi %30, %31 : index
                    %33 = arith.addi %30, %c1 : index
                    %34 = arith.addi %33, %31 : index
                    loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%29, %32], LR : [%29, %34]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %35 = loom.bufferize_to_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                    %36 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %37 = loom.semaphore_take %36 : memref<64x32xf16> -> memref<64x32xf16>
                    %38 = loom.init_tensor %37[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %39 = loom.semaphore_take %36 : memref<64x32xf16> -> memref<64x32xf16>
                    %40 = loom.init_tensor %39[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %41 = loom.broadcast ins(%35 : tensor<64xf16>) outs(%40 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                    %42 = arith.muli %arg12, %c1024 : index
                    %43 = arith.addi %24, %42 : index
                    %44 = arith.divui %22, %c64 : index
                    %45 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %46 = loom.semaphore_take %45 : memref<64x64xf16> -> memref<64x64xf16>
                    %c0_0 = arith.constant 0 : index
                    %47 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %43, %44, %c0_0)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                    loom.copy %reinterpret_cast_1, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%29, %32], LR : [%29, %34]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                    %48 = loom.bufferize_to_tensor %46[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %49 = arith.muli %arg10, %c32 : index
                    %50 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %51 = loom.semaphore_take %50 : memref<64x32xf16> -> memref<64x32xf16>
                    %c0_2 = arith.constant 0 : index
                    %52 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %23, %22, %c0_2, %49)
                    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%52], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                    %53 = arith.addi %28, %c1 : index
                    %54 = arith.addi %arg10, %30 : index
                    %55 = arith.addi %54, %31 : index
                    loom.copy %reinterpret_cast_3, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%28, %55], LR : [%53, %55]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                    %56 = loom.bufferize_to_tensor %51[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %57 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %58 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %59 = loom.init_tensor %58[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %60 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %61 = loom.init_tensor %60[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %62 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %63 = loom.init_tensor %62[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %64 = linalg.fill ins(%cst : f16) outs(%61 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %65 = linalg.matmul ins(%48, %56 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%64 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %51 : memref<64x32xf16>
                    loom.semaphore_give %46 : memref<64x64xf16>
                    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %41 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%63 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %out: f16):
                      %101 = math.exp %in_7 : f16
                      %102 = arith.mulf %in, %101 : f16
                      linalg.yield %102 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %60 : memref<64x32xf16>
                    loom.semaphore_give %39 : memref<64x32xf16>
                    %67 = arith.addi %21, %c1 : index
                    %68 = arith.muli %67, %c64 : index
                    %69 = arith.ceildivui %68, %c64 : index
                    %70 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %71 = loom.semaphore_take %70 : memref<64x32xf16> -> memref<64x32xf16>
                    %72 = loom.init_tensor %71[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %73 = loom.semaphore_take %70 : memref<64x32xf16> -> memref<64x32xf16>
                    %74 = loom.init_tensor %73[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %75 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %76 = loom.semaphore_take %75 : memref<64x64xf16> -> memref<64x64xf16>
                    %77 = loom.init_tensor %76[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %78 = loom.alloc [64] on @L1 : memref<64xf16>
                    %79 = loom.semaphore_take %78 : memref<64xf16> -> memref<64xf16>
                    %80 = loom.semaphore_take %78 : memref<64xf16> -> memref<64xf16>
                    %81 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %82 = loom.semaphore_take %81 : memref<32x64xf16> -> memref<32x64xf16>
                    %83 = loom.init_tensor %82[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %84 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %85 = loom.semaphore_take %84 : memref<32x64xf16> -> memref<32x64xf16>
                    %86 = loom.init_tensor %85[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %87 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %88 = loom.semaphore_take %87 : memref<64x32xf16> -> memref<64x32xf16>
                    %89 = scf.for %arg15 = %c0 to %69 step %c1 iter_args(%arg16 = %66) -> (tensor<64x32xf16>) {
                      %101 = arith.muli %arg15, %c64 : index
                      %102 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %23, %44, %24, %101)
                      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                      loom.copy %reinterpret_cast_7, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%29, %55], LR : [%29, %55]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                      %103 = loom.bufferize_to_tensor %76[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                      %104 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %101)
                      %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%104], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_8, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%29, %55], LR : [%29, %55]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %105 = loom.bufferize_to_tensor %80[64] : memref<64xf16> -> tensor<64xf16>
                      %106 = loom.broadcast ins(%35 : tensor<64xf16>) outs(%38 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                      %107 = loom.broadcast ins(%105 : tensor<64xf16>) outs(%83 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %80 : memref<64xf16>
                      %108 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %101)
                      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%108], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_9, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%29, %55], LR : [%29, %55]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %109 = loom.bufferize_to_tensor %79[64] : memref<64xf16> -> tensor<64xf16>
                      %110 = loom.broadcast ins(%109 : tensor<64xf16>) outs(%86 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %79 : memref<64xf16>
                      %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%103, %106, %107, %110 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%77 : tensor<64x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                        %118 = arith.subf %in_11, %in_12 : f16
                        %119 = math.exp %118 : f16
                        %120 = arith.mulf %in, %119 : f16
                        %121 = arith.mulf %120, %in_13 : f16
                        linalg.yield %121 : f16
                      } -> tensor<64x64xf16>
                      loom.semaphore_give %85 : memref<32x64xf16>
                      loom.semaphore_give %82 : memref<32x64xf16>
                      loom.semaphore_give %37 : memref<64x32xf16>
                      %112 = arith.addi %101, %42 : index
                      %113 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %112, %22, %49)
                      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                      loom.copy %reinterpret_cast_10, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%29, %55], LR : [%29, %55]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                      %114 = loom.bufferize_to_tensor %88[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                      %115 = linalg.fill ins(%cst : f16) outs(%74 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %116 = linalg.matmul ins(%111, %114 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%115 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %88 : memref<64x32xf16>
                      loom.semaphore_give %76 : memref<64x64xf16>
                      %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %116 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg16 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %118 = arith.addf %in, %in_11 : f16
                        linalg.yield %118 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %73 : memref<64x32xf16>
                      scf.yield %117 : tensor<64x32xf16>
                    } {loom.iter_type = #loom.iter_type<sequential>}
                    loom.semaphore_give %26 : memref<64xf16>
                    %90 = loom.alloc [1] on @L1 : memref<f16>
                    %91 = loom.semaphore_take %90 : memref<f16> -> memref<f16>
                    %92 = affine.apply affine_map<(d0) -> (d0)>(%22)
                    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%92], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                    %93 = arith.addi %31, %c3 : index
                    loom.copy %reinterpret_cast_4, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %31], LR : [%c7, %93]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                    %94 = loom.bufferize_to_tensor %91[] : memref<f16> -> tensor<f16>
                    %95 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %43, %22, %49)
                    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%95], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_5, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%29, %55], LR : [%29, %55]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %96 = loom.bufferize_to_tensor %71[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%89, %96, %94 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%72 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                      %101 = arith.mulf %in_7, %in_8 : f16
                      %102 = arith.addf %in, %101 : f16
                      linalg.yield %102 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %91 : memref<f16>
                    loom.semaphore_give %62 : memref<64x32xf16>
                    %98 = loom.sync ins(%97 : tensor<64x32xf16>) outs(%59 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %71 : memref<64x32xf16>
                    %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %43, %22, %49)
                    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%99], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    %100 = loom.bufferize_to_memref %98 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %100, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%29, %55], LR : [%29, %55]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.semaphore_give %58 : memref<64x32xf16>
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc4_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (4) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c4 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                  %21 = arith.muli %20, %c8 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %arg9, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg11, %c4 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg12, %c2 : index
                  %30 = arith.muli %arg8, %c4 : index
                  %31 = arith.addi %29, %30 : index
                  %32 = arith.addi %29, %c1 : index
                  %33 = arith.addi %32, %30 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%28, %31], LR : [%28, %33]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %34 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %35 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %36 = loom.semaphore_take %35 : memref<64x32xf16> -> memref<64x32xf16>
                  %37 = loom.init_tensor %36[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %38 = loom.semaphore_take %35 : memref<64x32xf16> -> memref<64x32xf16>
                  %39 = loom.init_tensor %38[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %40 = loom.broadcast ins(%34 : tensor<64xf16>) outs(%39 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                  %41 = arith.muli %arg12, %c1024 : index
                  %42 = arith.addi %23, %41 : index
                  %43 = arith.divui %21, %c64 : index
                  %44 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %45 = loom.semaphore_take %44 : memref<64x64xf16> -> memref<64x64xf16>
                  %c0_0 = arith.constant 0 : index
                  %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %42, %43, %c0_0)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                  loom.copy %reinterpret_cast_1, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%28, %31], LR : [%28, %33]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                  %47 = loom.bufferize_to_tensor %45[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %48 = arith.muli %arg10, %c32 : index
                  %49 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %50 = loom.semaphore_take %49 : memref<64x32xf16> -> memref<64x32xf16>
                  %c0_2 = arith.constant 0 : index
                  %51 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %22, %21, %c0_2, %48)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%51], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                  %52 = arith.addi %27, %c3 : index
                  %53 = arith.addi %arg10, %29 : index
                  %54 = arith.addi %53, %30 : index
                  loom.copy %reinterpret_cast_3, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%27, %54], LR : [%52, %54]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                  %55 = loom.bufferize_to_tensor %50[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %56 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %57 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
                  %58 = loom.init_tensor %57[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %59 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
                  %60 = loom.init_tensor %59[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %61 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
                  %62 = loom.init_tensor %61[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %63 = linalg.fill ins(%cst : f16) outs(%60 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %64 = linalg.matmul ins(%47, %55 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%63 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %50 : memref<64x32xf16>
                  loom.semaphore_give %45 : memref<64x64xf16>
                  %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %40 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%62 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_7: f16, %out: f16):
                    %100 = math.exp %in_7 : f16
                    %101 = arith.mulf %in, %100 : f16
                    linalg.yield %101 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %59 : memref<64x32xf16>
                  loom.semaphore_give %38 : memref<64x32xf16>
                  %66 = arith.addi %arg9, %c1 : index
                  %67 = arith.muli %66, %c64 : index
                  %68 = arith.ceildivui %67, %c64 : index
                  %69 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %70 = loom.semaphore_take %69 : memref<64x32xf16> -> memref<64x32xf16>
                  %71 = loom.init_tensor %70[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %72 = loom.semaphore_take %69 : memref<64x32xf16> -> memref<64x32xf16>
                  %73 = loom.init_tensor %72[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %74 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %75 = loom.semaphore_take %74 : memref<64x64xf16> -> memref<64x64xf16>
                  %76 = loom.init_tensor %75[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %77 = loom.alloc [64] on @L1 : memref<64xf16>
                  %78 = loom.semaphore_take %77 : memref<64xf16> -> memref<64xf16>
                  %79 = loom.semaphore_take %77 : memref<64xf16> -> memref<64xf16>
                  %80 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %81 = loom.semaphore_take %80 : memref<32x64xf16> -> memref<32x64xf16>
                  %82 = loom.init_tensor %81[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %83 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %84 = loom.semaphore_take %83 : memref<32x64xf16> -> memref<32x64xf16>
                  %85 = loom.init_tensor %84[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %86 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %87 = loom.semaphore_take %86 : memref<64x32xf16> -> memref<64x32xf16>
                  %88 = scf.for %arg14 = %c0 to %68 step %c1 iter_args(%arg15 = %65) -> (tensor<64x32xf16>) {
                    %100 = arith.muli %arg14, %c64 : index
                    %101 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %43, %23, %100)
                    %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%101], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_7, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %54], LR : [%28, %54]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %102 = loom.bufferize_to_tensor %75[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %103 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %100)
                    %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_8, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %54], LR : [%28, %54]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %104 = loom.bufferize_to_tensor %79[64] : memref<64xf16> -> tensor<64xf16>
                    %105 = loom.broadcast ins(%34 : tensor<64xf16>) outs(%37 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                    %106 = loom.broadcast ins(%104 : tensor<64xf16>) outs(%82 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                    loom.semaphore_give %79 : memref<64xf16>
                    %107 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %100)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%107], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %54], LR : [%28, %54]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %108 = loom.bufferize_to_tensor %78[64] : memref<64xf16> -> tensor<64xf16>
                    %109 = loom.broadcast ins(%108 : tensor<64xf16>) outs(%85 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                    loom.semaphore_give %78 : memref<64xf16>
                    %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%102, %105, %106, %109 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%76 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                      %117 = arith.subf %in_11, %in_12 : f16
                      %118 = math.exp %117 : f16
                      %119 = arith.mulf %in, %118 : f16
                      %120 = arith.mulf %119, %in_13 : f16
                      linalg.yield %120 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %84 : memref<32x64xf16>
                    loom.semaphore_give %81 : memref<32x64xf16>
                    loom.semaphore_give %36 : memref<64x32xf16>
                    %111 = arith.addi %100, %41 : index
                    %112 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %111, %21, %48)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%112], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %87 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %54], LR : [%28, %54]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %113 = loom.bufferize_to_tensor %87[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %114 = linalg.fill ins(%cst : f16) outs(%73 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %115 = linalg.matmul ins(%110, %113 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%114 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %87 : memref<64x32xf16>
                    loom.semaphore_give %75 : memref<64x64xf16>
                    %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %115 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_11: f16, %out: f16):
                      %117 = arith.addf %in, %in_11 : f16
                      linalg.yield %117 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %72 : memref<64x32xf16>
                    scf.yield %116 : tensor<64x32xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %25 : memref<64xf16>
                  %89 = loom.alloc [1] on @L1 : memref<f16>
                  %90 = loom.semaphore_take %89 : memref<f16> -> memref<f16>
                  %91 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%91], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                  %92 = arith.addi %30, %c3 : index
                  loom.copy %reinterpret_cast_4, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %30], LR : [%c7, %92]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %93 = loom.bufferize_to_tensor %90[] : memref<f16> -> tensor<f16>
                  %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %42, %21, %48)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%94], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %reinterpret_cast_5, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %54], LR : [%28, %54]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                  %95 = loom.bufferize_to_tensor %70[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%88, %95, %93 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%71 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                    %100 = arith.mulf %in_7, %in_8 : f16
                    %101 = arith.addf %in, %100 : f16
                    linalg.yield %101 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %90 : memref<f16>
                  loom.semaphore_give %61 : memref<64x32xf16>
                  %97 = loom.sync ins(%96 : tensor<64x32xf16>) outs(%58 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %70 : memref<64x32xf16>
                  %98 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %42, %21, %48)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%98], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  %99 = loom.bufferize_to_memref %97 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %99, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%28, %54], LR : [%28, %54]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.semaphore_give %57 : memref<64x32xf16>
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc4_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (4) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c4 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                  %21 = arith.muli %20, %c8 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %arg9, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg12, %c4 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg11, %c2 : index
                  %30 = arith.muli %arg8, %c4 : index
                  %31 = arith.addi %29, %30 : index
                  %32 = arith.addi %29, %c1 : index
                  %33 = arith.addi %32, %30 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%28, %31], LR : [%28, %33]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %34 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %35 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %36 = loom.semaphore_take %35 : memref<64x32xf16> -> memref<64x32xf16>
                  %37 = loom.init_tensor %36[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %38 = loom.semaphore_take %35 : memref<64x32xf16> -> memref<64x32xf16>
                  %39 = loom.init_tensor %38[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %40 = loom.broadcast ins(%34 : tensor<64xf16>) outs(%39 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                  %41 = arith.muli %arg12, %c1024 : index
                  %42 = arith.addi %23, %41 : index
                  %43 = arith.divui %21, %c64 : index
                  %44 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %45 = loom.semaphore_take %44 : memref<64x64xf16> -> memref<64x64xf16>
                  %c0_0 = arith.constant 0 : index
                  %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %42, %43, %c0_0)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                  loom.copy %reinterpret_cast_1, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%28, %31], LR : [%28, %33]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                  %47 = loom.bufferize_to_tensor %45[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %48 = arith.muli %arg10, %c32 : index
                  %49 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %50 = loom.semaphore_take %49 : memref<64x32xf16> -> memref<64x32xf16>
                  %c0_2 = arith.constant 0 : index
                  %51 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %22, %21, %c0_2, %48)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%51], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                  %52 = arith.addi %27, %c3 : index
                  %53 = arith.addi %arg10, %29 : index
                  %54 = arith.addi %53, %30 : index
                  loom.copy %reinterpret_cast_3, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%27, %54], LR : [%52, %54]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                  %55 = loom.bufferize_to_tensor %50[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %56 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %57 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
                  %58 = loom.init_tensor %57[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %59 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
                  %60 = loom.init_tensor %59[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %61 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
                  %62 = loom.init_tensor %61[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %63 = linalg.fill ins(%cst : f16) outs(%60 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %64 = linalg.matmul ins(%47, %55 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%63 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %50 : memref<64x32xf16>
                  loom.semaphore_give %45 : memref<64x64xf16>
                  %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %40 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%62 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_7: f16, %out: f16):
                    %100 = math.exp %in_7 : f16
                    %101 = arith.mulf %in, %100 : f16
                    linalg.yield %101 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %59 : memref<64x32xf16>
                  loom.semaphore_give %38 : memref<64x32xf16>
                  %66 = arith.addi %arg9, %c1 : index
                  %67 = arith.muli %66, %c64 : index
                  %68 = arith.ceildivui %67, %c64 : index
                  %69 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %70 = loom.semaphore_take %69 : memref<64x32xf16> -> memref<64x32xf16>
                  %71 = loom.init_tensor %70[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %72 = loom.semaphore_take %69 : memref<64x32xf16> -> memref<64x32xf16>
                  %73 = loom.init_tensor %72[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %74 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %75 = loom.semaphore_take %74 : memref<64x64xf16> -> memref<64x64xf16>
                  %76 = loom.init_tensor %75[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %77 = loom.alloc [64] on @L1 : memref<64xf16>
                  %78 = loom.semaphore_take %77 : memref<64xf16> -> memref<64xf16>
                  %79 = loom.semaphore_take %77 : memref<64xf16> -> memref<64xf16>
                  %80 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %81 = loom.semaphore_take %80 : memref<32x64xf16> -> memref<32x64xf16>
                  %82 = loom.init_tensor %81[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %83 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %84 = loom.semaphore_take %83 : memref<32x64xf16> -> memref<32x64xf16>
                  %85 = loom.init_tensor %84[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %86 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %87 = loom.semaphore_take %86 : memref<64x32xf16> -> memref<64x32xf16>
                  %88 = scf.for %arg14 = %c0 to %68 step %c1 iter_args(%arg15 = %65) -> (tensor<64x32xf16>) {
                    %100 = arith.muli %arg14, %c64 : index
                    %101 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %43, %23, %100)
                    %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%101], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_7, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %54], LR : [%28, %54]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %102 = loom.bufferize_to_tensor %75[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %103 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %100)
                    %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%103], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_8, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %54], LR : [%28, %54]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %104 = loom.bufferize_to_tensor %79[64] : memref<64xf16> -> tensor<64xf16>
                    %105 = loom.broadcast ins(%34 : tensor<64xf16>) outs(%37 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                    %106 = loom.broadcast ins(%104 : tensor<64xf16>) outs(%82 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                    loom.semaphore_give %79 : memref<64xf16>
                    %107 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %100)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%107], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %54], LR : [%28, %54]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %108 = loom.bufferize_to_tensor %78[64] : memref<64xf16> -> tensor<64xf16>
                    %109 = loom.broadcast ins(%108 : tensor<64xf16>) outs(%85 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                    loom.semaphore_give %78 : memref<64xf16>
                    %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%102, %105, %106, %109 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%76 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                      %117 = arith.subf %in_11, %in_12 : f16
                      %118 = math.exp %117 : f16
                      %119 = arith.mulf %in, %118 : f16
                      %120 = arith.mulf %119, %in_13 : f16
                      linalg.yield %120 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %84 : memref<32x64xf16>
                    loom.semaphore_give %81 : memref<32x64xf16>
                    loom.semaphore_give %36 : memref<64x32xf16>
                    %111 = arith.addi %100, %41 : index
                    %112 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %111, %21, %48)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%112], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %87 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %54], LR : [%28, %54]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %113 = loom.bufferize_to_tensor %87[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %114 = linalg.fill ins(%cst : f16) outs(%73 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %115 = linalg.matmul ins(%110, %113 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%114 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %87 : memref<64x32xf16>
                    loom.semaphore_give %75 : memref<64x64xf16>
                    %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %115 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_11: f16, %out: f16):
                      %117 = arith.addf %in, %in_11 : f16
                      linalg.yield %117 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %72 : memref<64x32xf16>
                    scf.yield %116 : tensor<64x32xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %25 : memref<64xf16>
                  %89 = loom.alloc [1] on @L1 : memref<f16>
                  %90 = loom.semaphore_take %89 : memref<f16> -> memref<f16>
                  %91 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%91], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                  %92 = arith.addi %30, %c3 : index
                  loom.copy %reinterpret_cast_4, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %30], LR : [%c7, %92]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %93 = loom.bufferize_to_tensor %90[] : memref<f16> -> tensor<f16>
                  %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %42, %21, %48)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%94], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %reinterpret_cast_5, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%28, %54], LR : [%28, %54]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                  %95 = loom.bufferize_to_tensor %70[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%88, %95, %93 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%71 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                    %100 = arith.mulf %in_7, %in_8 : f16
                    %101 = arith.addf %in, %100 : f16
                    linalg.yield %101 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %90 : memref<f16>
                  loom.semaphore_give %61 : memref<64x32xf16>
                  %97 = loom.sync ins(%96 : tensor<64x32xf16>) outs(%58 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %70 : memref<64x32xf16>
                  %98 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %42, %21, %48)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%98], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  %99 = loom.bufferize_to_memref %97 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %99, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%28, %54], LR : [%28, %54]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.semaphore_give %57 : memref<64x32xf16>
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (4) {
                scf.for %arg13 = %c0 to %c4 step %c1 {
                  scf.for %arg14 = %c0 to %c2 step %c1 {
                    %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                    %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                    %22 = arith.muli %20, %c8 : index
                    %23 = arith.muli %arg12, %c4 : index
                    %24 = arith.muli %21, %c64 : index
                    %25 = loom.alloc [64] on @L1 : memref<64xf16>
                    %26 = loom.semaphore_take %25 : memref<64xf16> -> memref<64xf16>
                    %27 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    %28 = arith.muli %arg11, %c2 : index
                    %29 = arith.addi %arg9, %28 : index
                    %30 = arith.muli %arg8, %c4 : index
                    %31 = arith.addi %29, %30 : index
                    %32 = arith.muli %arg12, %c2 : index
                    %33 = arith.addi %32, %c1 : index
                    loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %32], LR : [%31, %33]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %34 = loom.bufferize_to_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                    %35 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %36 = loom.semaphore_take %35 : memref<64x32xf16> -> memref<64x32xf16>
                    %37 = loom.init_tensor %36[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %38 = loom.semaphore_take %35 : memref<64x32xf16> -> memref<64x32xf16>
                    %39 = loom.init_tensor %38[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %40 = loom.broadcast ins(%34 : tensor<64xf16>) outs(%39 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                    %41 = arith.muli %arg12, %c1024 : index
                    %42 = arith.addi %24, %41 : index
                    %43 = arith.divui %22, %c64 : index
                    %44 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %45 = loom.semaphore_take %44 : memref<64x64xf16> -> memref<64x64xf16>
                    %c0_0 = arith.constant 0 : index
                    %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %42, %43, %c0_0)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                    loom.copy %reinterpret_cast_1, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %32], LR : [%31, %33]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                    %47 = loom.bufferize_to_tensor %45[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %48 = arith.muli %arg10, %c32 : index
                    %49 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %50 = loom.semaphore_take %49 : memref<64x32xf16> -> memref<64x32xf16>
                    %c0_2 = arith.constant 0 : index
                    %51 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %23, %22, %c0_2, %48)
                    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%51], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                    %52 = arith.addi %28, %30 : index
                    %53 = arith.addi %28, %c1 : index
                    %54 = arith.addi %53, %30 : index
                    %55 = arith.addi %arg10, %32 : index
                    loom.copy %reinterpret_cast_3, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%52, %55], LR : [%54, %55]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                    %56 = loom.bufferize_to_tensor %50[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %57 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %58 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %59 = loom.init_tensor %58[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %60 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %61 = loom.init_tensor %60[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %62 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %63 = loom.init_tensor %62[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %64 = linalg.fill ins(%cst : f16) outs(%61 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %65 = linalg.matmul ins(%47, %56 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%64 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %50 : memref<64x32xf16>
                    loom.semaphore_give %45 : memref<64x64xf16>
                    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %40 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%63 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %out: f16):
                      %101 = math.exp %in_7 : f16
                      %102 = arith.mulf %in, %101 : f16
                      linalg.yield %102 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %60 : memref<64x32xf16>
                    loom.semaphore_give %38 : memref<64x32xf16>
                    %67 = arith.addi %21, %c1 : index
                    %68 = arith.muli %67, %c64 : index
                    %69 = arith.ceildivui %68, %c64 : index
                    %70 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %71 = loom.semaphore_take %70 : memref<64x32xf16> -> memref<64x32xf16>
                    %72 = loom.init_tensor %71[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %73 = loom.semaphore_take %70 : memref<64x32xf16> -> memref<64x32xf16>
                    %74 = loom.init_tensor %73[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %75 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %76 = loom.semaphore_take %75 : memref<64x64xf16> -> memref<64x64xf16>
                    %77 = loom.init_tensor %76[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %78 = loom.alloc [64] on @L1 : memref<64xf16>
                    %79 = loom.semaphore_take %78 : memref<64xf16> -> memref<64xf16>
                    %80 = loom.semaphore_take %78 : memref<64xf16> -> memref<64xf16>
                    %81 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %82 = loom.semaphore_take %81 : memref<32x64xf16> -> memref<32x64xf16>
                    %83 = loom.init_tensor %82[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %84 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %85 = loom.semaphore_take %84 : memref<32x64xf16> -> memref<32x64xf16>
                    %86 = loom.init_tensor %85[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %87 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %88 = loom.semaphore_take %87 : memref<64x32xf16> -> memref<64x32xf16>
                    %89 = scf.for %arg15 = %c0 to %69 step %c1 iter_args(%arg16 = %66) -> (tensor<64x32xf16>) {
                      %101 = arith.muli %arg15, %c64 : index
                      %102 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %23, %43, %24, %101)
                      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                      loom.copy %reinterpret_cast_7, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                      %103 = loom.bufferize_to_tensor %76[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                      %104 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %101)
                      %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%104], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_8, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %105 = loom.bufferize_to_tensor %80[64] : memref<64xf16> -> tensor<64xf16>
                      %106 = loom.broadcast ins(%34 : tensor<64xf16>) outs(%37 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                      %107 = loom.broadcast ins(%105 : tensor<64xf16>) outs(%83 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %80 : memref<64xf16>
                      %108 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %101)
                      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%108], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_9, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %109 = loom.bufferize_to_tensor %79[64] : memref<64xf16> -> tensor<64xf16>
                      %110 = loom.broadcast ins(%109 : tensor<64xf16>) outs(%86 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %79 : memref<64xf16>
                      %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%103, %106, %107, %110 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%77 : tensor<64x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                        %118 = arith.subf %in_11, %in_12 : f16
                        %119 = math.exp %118 : f16
                        %120 = arith.mulf %in, %119 : f16
                        %121 = arith.mulf %120, %in_13 : f16
                        linalg.yield %121 : f16
                      } -> tensor<64x64xf16>
                      loom.semaphore_give %85 : memref<32x64xf16>
                      loom.semaphore_give %82 : memref<32x64xf16>
                      loom.semaphore_give %36 : memref<64x32xf16>
                      %112 = arith.addi %101, %41 : index
                      %113 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %112, %22, %48)
                      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                      loom.copy %reinterpret_cast_10, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                      %114 = loom.bufferize_to_tensor %88[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                      %115 = linalg.fill ins(%cst : f16) outs(%74 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %116 = linalg.matmul ins(%111, %114 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%115 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %88 : memref<64x32xf16>
                      loom.semaphore_give %76 : memref<64x64xf16>
                      %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %116 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg16 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %118 = arith.addf %in, %in_11 : f16
                        linalg.yield %118 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %73 : memref<64x32xf16>
                      scf.yield %117 : tensor<64x32xf16>
                    } {loom.iter_type = #loom.iter_type<sequential>}
                    loom.semaphore_give %26 : memref<64xf16>
                    %90 = loom.alloc [1] on @L1 : memref<f16>
                    %91 = loom.semaphore_take %90 : memref<f16> -> memref<f16>
                    %92 = affine.apply affine_map<(d0) -> (d0)>(%22)
                    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%92], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                    %93 = arith.addi %30, %c3 : index
                    loom.copy %reinterpret_cast_4, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%30, %c0], LR : [%93, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                    %94 = loom.bufferize_to_tensor %91[] : memref<f16> -> tensor<f16>
                    %95 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %42, %22, %48)
                    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%95], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_5, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %96 = loom.bufferize_to_tensor %71[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%89, %96, %94 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%72 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                      %101 = arith.mulf %in_7, %in_8 : f16
                      %102 = arith.addf %in, %101 : f16
                      linalg.yield %102 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %91 : memref<f16>
                    loom.semaphore_give %62 : memref<64x32xf16>
                    %98 = loom.sync ins(%97 : tensor<64x32xf16>) outs(%59 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %71 : memref<64x32xf16>
                    %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %42, %22, %48)
                    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%99], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    %100 = loom.bufferize_to_memref %98 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %100, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.semaphore_give %58 : memref<64x32xf16>
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (4) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c4 step %c1 {
                  scf.for %arg14 = %c0 to %c2 step %c1 {
                    %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                    %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                    %22 = arith.muli %20, %c8 : index
                    %23 = arith.muli %arg12, %c4 : index
                    %24 = arith.muli %21, %c64 : index
                    %25 = loom.alloc [64] on @L1 : memref<64xf16>
                    %26 = loom.semaphore_take %25 : memref<64xf16> -> memref<64xf16>
                    %27 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    %28 = arith.muli %arg12, %c2 : index
                    %29 = arith.addi %arg9, %28 : index
                    %30 = arith.muli %arg8, %c4 : index
                    %31 = arith.addi %29, %30 : index
                    %32 = arith.muli %arg11, %c2 : index
                    %33 = arith.addi %32, %c1 : index
                    loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %32], LR : [%31, %33]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %34 = loom.bufferize_to_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                    %35 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %36 = loom.semaphore_take %35 : memref<64x32xf16> -> memref<64x32xf16>
                    %37 = loom.init_tensor %36[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %38 = loom.semaphore_take %35 : memref<64x32xf16> -> memref<64x32xf16>
                    %39 = loom.init_tensor %38[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %40 = loom.broadcast ins(%34 : tensor<64xf16>) outs(%39 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                    %41 = arith.muli %arg12, %c1024 : index
                    %42 = arith.addi %24, %41 : index
                    %43 = arith.divui %22, %c64 : index
                    %44 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %45 = loom.semaphore_take %44 : memref<64x64xf16> -> memref<64x64xf16>
                    %c0_0 = arith.constant 0 : index
                    %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %42, %43, %c0_0)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                    loom.copy %reinterpret_cast_1, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %32], LR : [%31, %33]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                    %47 = loom.bufferize_to_tensor %45[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %48 = arith.muli %arg10, %c32 : index
                    %49 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %50 = loom.semaphore_take %49 : memref<64x32xf16> -> memref<64x32xf16>
                    %c0_2 = arith.constant 0 : index
                    %51 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %23, %22, %c0_2, %48)
                    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%51], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                    %52 = arith.addi %28, %30 : index
                    %53 = arith.addi %28, %c1 : index
                    %54 = arith.addi %53, %30 : index
                    %55 = arith.addi %arg10, %32 : index
                    loom.copy %reinterpret_cast_3, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%52, %55], LR : [%54, %55]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                    %56 = loom.bufferize_to_tensor %50[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %57 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %58 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %59 = loom.init_tensor %58[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %60 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %61 = loom.init_tensor %60[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %62 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %63 = loom.init_tensor %62[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %64 = linalg.fill ins(%cst : f16) outs(%61 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %65 = linalg.matmul ins(%47, %56 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%64 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %50 : memref<64x32xf16>
                    loom.semaphore_give %45 : memref<64x64xf16>
                    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %40 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%63 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %out: f16):
                      %101 = math.exp %in_7 : f16
                      %102 = arith.mulf %in, %101 : f16
                      linalg.yield %102 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %60 : memref<64x32xf16>
                    loom.semaphore_give %38 : memref<64x32xf16>
                    %67 = arith.addi %21, %c1 : index
                    %68 = arith.muli %67, %c64 : index
                    %69 = arith.ceildivui %68, %c64 : index
                    %70 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %71 = loom.semaphore_take %70 : memref<64x32xf16> -> memref<64x32xf16>
                    %72 = loom.init_tensor %71[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %73 = loom.semaphore_take %70 : memref<64x32xf16> -> memref<64x32xf16>
                    %74 = loom.init_tensor %73[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %75 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %76 = loom.semaphore_take %75 : memref<64x64xf16> -> memref<64x64xf16>
                    %77 = loom.init_tensor %76[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %78 = loom.alloc [64] on @L1 : memref<64xf16>
                    %79 = loom.semaphore_take %78 : memref<64xf16> -> memref<64xf16>
                    %80 = loom.semaphore_take %78 : memref<64xf16> -> memref<64xf16>
                    %81 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %82 = loom.semaphore_take %81 : memref<32x64xf16> -> memref<32x64xf16>
                    %83 = loom.init_tensor %82[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %84 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %85 = loom.semaphore_take %84 : memref<32x64xf16> -> memref<32x64xf16>
                    %86 = loom.init_tensor %85[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %87 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %88 = loom.semaphore_take %87 : memref<64x32xf16> -> memref<64x32xf16>
                    %89 = scf.for %arg15 = %c0 to %69 step %c1 iter_args(%arg16 = %66) -> (tensor<64x32xf16>) {
                      %101 = arith.muli %arg15, %c64 : index
                      %102 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %23, %43, %24, %101)
                      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                      loom.copy %reinterpret_cast_7, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                      %103 = loom.bufferize_to_tensor %76[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                      %104 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %101)
                      %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%104], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_8, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %105 = loom.bufferize_to_tensor %80[64] : memref<64xf16> -> tensor<64xf16>
                      %106 = loom.broadcast ins(%34 : tensor<64xf16>) outs(%37 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                      %107 = loom.broadcast ins(%105 : tensor<64xf16>) outs(%83 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %80 : memref<64xf16>
                      %108 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %101)
                      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%108], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_9, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %109 = loom.bufferize_to_tensor %79[64] : memref<64xf16> -> tensor<64xf16>
                      %110 = loom.broadcast ins(%109 : tensor<64xf16>) outs(%86 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %79 : memref<64xf16>
                      %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%103, %106, %107, %110 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%77 : tensor<64x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                        %118 = arith.subf %in_11, %in_12 : f16
                        %119 = math.exp %118 : f16
                        %120 = arith.mulf %in, %119 : f16
                        %121 = arith.mulf %120, %in_13 : f16
                        linalg.yield %121 : f16
                      } -> tensor<64x64xf16>
                      loom.semaphore_give %85 : memref<32x64xf16>
                      loom.semaphore_give %82 : memref<32x64xf16>
                      loom.semaphore_give %36 : memref<64x32xf16>
                      %112 = arith.addi %101, %41 : index
                      %113 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %112, %22, %48)
                      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                      loom.copy %reinterpret_cast_10, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                      %114 = loom.bufferize_to_tensor %88[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                      %115 = linalg.fill ins(%cst : f16) outs(%74 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %116 = linalg.matmul ins(%111, %114 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%115 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %88 : memref<64x32xf16>
                      loom.semaphore_give %76 : memref<64x64xf16>
                      %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %116 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg16 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %118 = arith.addf %in, %in_11 : f16
                        linalg.yield %118 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %73 : memref<64x32xf16>
                      scf.yield %117 : tensor<64x32xf16>
                    } {loom.iter_type = #loom.iter_type<sequential>}
                    loom.semaphore_give %26 : memref<64xf16>
                    %90 = loom.alloc [1] on @L1 : memref<f16>
                    %91 = loom.semaphore_take %90 : memref<f16> -> memref<f16>
                    %92 = affine.apply affine_map<(d0) -> (d0)>(%22)
                    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%92], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                    %93 = arith.addi %30, %c3 : index
                    loom.copy %reinterpret_cast_4, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%30, %c0], LR : [%93, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                    %94 = loom.bufferize_to_tensor %91[] : memref<f16> -> tensor<f16>
                    %95 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %42, %22, %48)
                    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%95], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_5, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %96 = loom.bufferize_to_tensor %71[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%89, %96, %94 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%72 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                      %101 = arith.mulf %in_7, %in_8 : f16
                      %102 = arith.addf %in, %101 : f16
                      linalg.yield %102 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %91 : memref<f16>
                    loom.semaphore_give %62 : memref<64x32xf16>
                    %98 = loom.sync ins(%97 : tensor<64x32xf16>) outs(%59 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %71 : memref<64x32xf16>
                    %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %42, %22, %48)
                    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%99], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    %100 = loom.bufferize_to_memref %98 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %100, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.semaphore_give %58 : memref<64x32xf16>
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc4_dim_y_level0_bc4_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (4) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c4 step %c1 {
                  scf.for %arg14 = %c0 to %c2 step %c1 {
                    %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                    %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                    %22 = arith.muli %20, %c8 : index
                    %23 = arith.muli %arg12, %c4 : index
                    %24 = arith.muli %21, %c64 : index
                    %25 = loom.alloc [64] on @L1 : memref<64xf16>
                    %26 = loom.semaphore_take %25 : memref<64xf16> -> memref<64xf16>
                    %27 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    %28 = arith.muli %arg11, %c2 : index
                    %29 = arith.addi %arg9, %28 : index
                    %30 = arith.muli %arg8, %c4 : index
                    %31 = arith.addi %29, %30 : index
                    %32 = arith.muli %arg12, %c4 : index
                    %33 = arith.addi %32, %c3 : index
                    loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%31, %32], LR : [%31, %33]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %34 = loom.bufferize_to_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                    %35 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %36 = loom.semaphore_take %35 : memref<64x32xf16> -> memref<64x32xf16>
                    %37 = loom.init_tensor %36[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %38 = loom.semaphore_take %35 : memref<64x32xf16> -> memref<64x32xf16>
                    %39 = loom.init_tensor %38[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %40 = loom.broadcast ins(%34 : tensor<64xf16>) outs(%39 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                    %41 = arith.muli %arg12, %c1024 : index
                    %42 = arith.addi %24, %41 : index
                    %43 = arith.divui %22, %c64 : index
                    %44 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %45 = loom.semaphore_take %44 : memref<64x64xf16> -> memref<64x64xf16>
                    %c0_0 = arith.constant 0 : index
                    %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %42, %43, %c0_0)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                    loom.copy %reinterpret_cast_1, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%31, %32], LR : [%31, %33]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                    %47 = loom.bufferize_to_tensor %45[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %48 = arith.muli %arg10, %c32 : index
                    %49 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %50 = loom.semaphore_take %49 : memref<64x32xf16> -> memref<64x32xf16>
                    %c0_2 = arith.constant 0 : index
                    %51 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %23, %22, %c0_2, %48)
                    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%51], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                    %52 = arith.addi %28, %30 : index
                    %53 = arith.addi %28, %c1 : index
                    %54 = arith.addi %53, %30 : index
                    %55 = arith.addi %arg10, %32 : index
                    loom.copy %reinterpret_cast_3, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%52, %55], LR : [%54, %55]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                    %56 = loom.bufferize_to_tensor %50[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %57 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %58 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %59 = loom.init_tensor %58[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %60 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %61 = loom.init_tensor %60[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %62 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %63 = loom.init_tensor %62[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %64 = linalg.fill ins(%cst : f16) outs(%61 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %65 = linalg.matmul ins(%47, %56 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%64 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %50 : memref<64x32xf16>
                    loom.semaphore_give %45 : memref<64x64xf16>
                    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %40 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%63 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %out: f16):
                      %101 = math.exp %in_7 : f16
                      %102 = arith.mulf %in, %101 : f16
                      linalg.yield %102 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %60 : memref<64x32xf16>
                    loom.semaphore_give %38 : memref<64x32xf16>
                    %67 = arith.addi %21, %c1 : index
                    %68 = arith.muli %67, %c64 : index
                    %69 = arith.ceildivui %68, %c64 : index
                    %70 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %71 = loom.semaphore_take %70 : memref<64x32xf16> -> memref<64x32xf16>
                    %72 = loom.init_tensor %71[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %73 = loom.semaphore_take %70 : memref<64x32xf16> -> memref<64x32xf16>
                    %74 = loom.init_tensor %73[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %75 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %76 = loom.semaphore_take %75 : memref<64x64xf16> -> memref<64x64xf16>
                    %77 = loom.init_tensor %76[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %78 = loom.alloc [64] on @L1 : memref<64xf16>
                    %79 = loom.semaphore_take %78 : memref<64xf16> -> memref<64xf16>
                    %80 = loom.semaphore_take %78 : memref<64xf16> -> memref<64xf16>
                    %81 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %82 = loom.semaphore_take %81 : memref<32x64xf16> -> memref<32x64xf16>
                    %83 = loom.init_tensor %82[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %84 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %85 = loom.semaphore_take %84 : memref<32x64xf16> -> memref<32x64xf16>
                    %86 = loom.init_tensor %85[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %87 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %88 = loom.semaphore_take %87 : memref<64x32xf16> -> memref<64x32xf16>
                    %89 = scf.for %arg15 = %c0 to %69 step %c1 iter_args(%arg16 = %66) -> (tensor<64x32xf16>) {
                      %101 = arith.muli %arg15, %c64 : index
                      %102 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %23, %43, %24, %101)
                      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                      loom.copy %reinterpret_cast_7, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                      %103 = loom.bufferize_to_tensor %76[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                      %104 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %101)
                      %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%104], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_8, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %105 = loom.bufferize_to_tensor %80[64] : memref<64xf16> -> tensor<64xf16>
                      %106 = loom.broadcast ins(%34 : tensor<64xf16>) outs(%37 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                      %107 = loom.broadcast ins(%105 : tensor<64xf16>) outs(%83 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %80 : memref<64xf16>
                      %108 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %101)
                      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%108], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_9, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %109 = loom.bufferize_to_tensor %79[64] : memref<64xf16> -> tensor<64xf16>
                      %110 = loom.broadcast ins(%109 : tensor<64xf16>) outs(%86 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %79 : memref<64xf16>
                      %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%103, %106, %107, %110 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%77 : tensor<64x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                        %118 = arith.subf %in_11, %in_12 : f16
                        %119 = math.exp %118 : f16
                        %120 = arith.mulf %in, %119 : f16
                        %121 = arith.mulf %120, %in_13 : f16
                        linalg.yield %121 : f16
                      } -> tensor<64x64xf16>
                      loom.semaphore_give %85 : memref<32x64xf16>
                      loom.semaphore_give %82 : memref<32x64xf16>
                      loom.semaphore_give %36 : memref<64x32xf16>
                      %112 = arith.addi %101, %41 : index
                      %113 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %112, %22, %48)
                      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                      loom.copy %reinterpret_cast_10, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                      %114 = loom.bufferize_to_tensor %88[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                      %115 = linalg.fill ins(%cst : f16) outs(%74 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %116 = linalg.matmul ins(%111, %114 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%115 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %88 : memref<64x32xf16>
                      loom.semaphore_give %76 : memref<64x64xf16>
                      %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %116 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg16 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %118 = arith.addf %in, %in_11 : f16
                        linalg.yield %118 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %73 : memref<64x32xf16>
                      scf.yield %117 : tensor<64x32xf16>
                    } {loom.iter_type = #loom.iter_type<sequential>}
                    loom.semaphore_give %26 : memref<64xf16>
                    %90 = loom.alloc [1] on @L1 : memref<f16>
                    %91 = loom.semaphore_take %90 : memref<f16> -> memref<f16>
                    %92 = affine.apply affine_map<(d0) -> (d0)>(%22)
                    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%92], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                    %93 = arith.addi %30, %c3 : index
                    loom.copy %reinterpret_cast_4, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%30, %c0], LR : [%93, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                    %94 = loom.bufferize_to_tensor %91[] : memref<f16> -> tensor<f16>
                    %95 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %42, %22, %48)
                    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%95], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_5, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %96 = loom.bufferize_to_tensor %71[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%89, %96, %94 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%72 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                      %101 = arith.mulf %in_7, %in_8 : f16
                      %102 = arith.addf %in, %101 : f16
                      linalg.yield %102 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %91 : memref<f16>
                    loom.semaphore_give %62 : memref<64x32xf16>
                    %98 = loom.sync ins(%97 : tensor<64x32xf16>) outs(%59 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %71 : memref<64x32xf16>
                    %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %42, %22, %48)
                    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%99], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    %100 = loom.bufferize_to_memref %98 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %100, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.semaphore_give %58 : memref<64x32xf16>
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_y_level0_bc4_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (4) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c4 step %c1 {
                  scf.for %arg14 = %c0 to %c2 step %c1 {
                    %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                    %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                    %22 = arith.muli %20, %c8 : index
                    %23 = arith.muli %arg12, %c4 : index
                    %24 = arith.muli %21, %c64 : index
                    %25 = loom.alloc [64] on @L1 : memref<64xf16>
                    %26 = loom.semaphore_take %25 : memref<64xf16> -> memref<64xf16>
                    %27 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    %28 = arith.muli %arg12, %c2 : index
                    %29 = arith.addi %arg9, %28 : index
                    %30 = arith.muli %arg8, %c4 : index
                    %31 = arith.addi %29, %30 : index
                    %32 = arith.muli %arg11, %c4 : index
                    %33 = arith.addi %32, %c3 : index
                    loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%31, %32], LR : [%31, %33]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %34 = loom.bufferize_to_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                    %35 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %36 = loom.semaphore_take %35 : memref<64x32xf16> -> memref<64x32xf16>
                    %37 = loom.init_tensor %36[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %38 = loom.semaphore_take %35 : memref<64x32xf16> -> memref<64x32xf16>
                    %39 = loom.init_tensor %38[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %40 = loom.broadcast ins(%34 : tensor<64xf16>) outs(%39 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                    %41 = arith.muli %arg12, %c1024 : index
                    %42 = arith.addi %24, %41 : index
                    %43 = arith.divui %22, %c64 : index
                    %44 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %45 = loom.semaphore_take %44 : memref<64x64xf16> -> memref<64x64xf16>
                    %c0_0 = arith.constant 0 : index
                    %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %42, %43, %c0_0)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                    loom.copy %reinterpret_cast_1, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%31, %32], LR : [%31, %33]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                    %47 = loom.bufferize_to_tensor %45[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %48 = arith.muli %arg10, %c32 : index
                    %49 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %50 = loom.semaphore_take %49 : memref<64x32xf16> -> memref<64x32xf16>
                    %c0_2 = arith.constant 0 : index
                    %51 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %23, %22, %c0_2, %48)
                    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%51], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                    %52 = arith.addi %28, %30 : index
                    %53 = arith.addi %28, %c1 : index
                    %54 = arith.addi %53, %30 : index
                    %55 = arith.addi %arg10, %32 : index
                    loom.copy %reinterpret_cast_3, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%52, %55], LR : [%54, %55]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                    %56 = loom.bufferize_to_tensor %50[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %57 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %58 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %59 = loom.init_tensor %58[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %60 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %61 = loom.init_tensor %60[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %62 = loom.semaphore_take %57 : memref<64x32xf16> -> memref<64x32xf16>
                    %63 = loom.init_tensor %62[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %64 = linalg.fill ins(%cst : f16) outs(%61 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %65 = linalg.matmul ins(%47, %56 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%64 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %50 : memref<64x32xf16>
                    loom.semaphore_give %45 : memref<64x64xf16>
                    %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %40 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%63 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %out: f16):
                      %101 = math.exp %in_7 : f16
                      %102 = arith.mulf %in, %101 : f16
                      linalg.yield %102 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %60 : memref<64x32xf16>
                    loom.semaphore_give %38 : memref<64x32xf16>
                    %67 = arith.addi %21, %c1 : index
                    %68 = arith.muli %67, %c64 : index
                    %69 = arith.ceildivui %68, %c64 : index
                    %70 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %71 = loom.semaphore_take %70 : memref<64x32xf16> -> memref<64x32xf16>
                    %72 = loom.init_tensor %71[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %73 = loom.semaphore_take %70 : memref<64x32xf16> -> memref<64x32xf16>
                    %74 = loom.init_tensor %73[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %75 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %76 = loom.semaphore_take %75 : memref<64x64xf16> -> memref<64x64xf16>
                    %77 = loom.init_tensor %76[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %78 = loom.alloc [64] on @L1 : memref<64xf16>
                    %79 = loom.semaphore_take %78 : memref<64xf16> -> memref<64xf16>
                    %80 = loom.semaphore_take %78 : memref<64xf16> -> memref<64xf16>
                    %81 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %82 = loom.semaphore_take %81 : memref<32x64xf16> -> memref<32x64xf16>
                    %83 = loom.init_tensor %82[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %84 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %85 = loom.semaphore_take %84 : memref<32x64xf16> -> memref<32x64xf16>
                    %86 = loom.init_tensor %85[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %87 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %88 = loom.semaphore_take %87 : memref<64x32xf16> -> memref<64x32xf16>
                    %89 = scf.for %arg15 = %c0 to %69 step %c1 iter_args(%arg16 = %66) -> (tensor<64x32xf16>) {
                      %101 = arith.muli %arg15, %c64 : index
                      %102 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %23, %43, %24, %101)
                      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%102], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                      loom.copy %reinterpret_cast_7, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                      %103 = loom.bufferize_to_tensor %76[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                      %104 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %101)
                      %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%104], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_8, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %105 = loom.bufferize_to_tensor %80[64] : memref<64xf16> -> tensor<64xf16>
                      %106 = loom.broadcast ins(%34 : tensor<64xf16>) outs(%37 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                      %107 = loom.broadcast ins(%105 : tensor<64xf16>) outs(%83 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %80 : memref<64xf16>
                      %108 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %101)
                      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%108], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_9, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %109 = loom.bufferize_to_tensor %79[64] : memref<64xf16> -> tensor<64xf16>
                      %110 = loom.broadcast ins(%109 : tensor<64xf16>) outs(%86 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %79 : memref<64xf16>
                      %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%103, %106, %107, %110 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%77 : tensor<64x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                        %118 = arith.subf %in_11, %in_12 : f16
                        %119 = math.exp %118 : f16
                        %120 = arith.mulf %in, %119 : f16
                        %121 = arith.mulf %120, %in_13 : f16
                        linalg.yield %121 : f16
                      } -> tensor<64x64xf16>
                      loom.semaphore_give %85 : memref<32x64xf16>
                      loom.semaphore_give %82 : memref<32x64xf16>
                      loom.semaphore_give %36 : memref<64x32xf16>
                      %112 = arith.addi %101, %41 : index
                      %113 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %112, %22, %48)
                      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                      loom.copy %reinterpret_cast_10, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                      %114 = loom.bufferize_to_tensor %88[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                      %115 = linalg.fill ins(%cst : f16) outs(%74 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %116 = linalg.matmul ins(%111, %114 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%115 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %88 : memref<64x32xf16>
                      loom.semaphore_give %76 : memref<64x64xf16>
                      %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %116 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg16 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %118 = arith.addf %in, %in_11 : f16
                        linalg.yield %118 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %73 : memref<64x32xf16>
                      scf.yield %117 : tensor<64x32xf16>
                    } {loom.iter_type = #loom.iter_type<sequential>}
                    loom.semaphore_give %26 : memref<64xf16>
                    %90 = loom.alloc [1] on @L1 : memref<f16>
                    %91 = loom.semaphore_take %90 : memref<f16> -> memref<f16>
                    %92 = affine.apply affine_map<(d0) -> (d0)>(%22)
                    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%92], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                    %93 = arith.addi %30, %c3 : index
                    loom.copy %reinterpret_cast_4, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%30, %c0], LR : [%93, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                    %94 = loom.bufferize_to_tensor %91[] : memref<f16> -> tensor<f16>
                    %95 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %42, %22, %48)
                    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%95], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_5, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %96 = loom.bufferize_to_tensor %71[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%89, %96, %94 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%72 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                      %101 = arith.mulf %in_7, %in_8 : f16
                      %102 = arith.addf %in, %101 : f16
                      linalg.yield %102 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %91 : memref<f16>
                    loom.semaphore_give %62 : memref<64x32xf16>
                    %98 = loom.sync ins(%97 : tensor<64x32xf16>) outs(%59 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %71 : memref<64x32xf16>
                    %99 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %42, %22, %48)
                    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%99], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    %100 = loom.bufferize_to_memref %98 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %100, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %55], LR : [%31, %55]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.semaphore_give %58 : memref<64x32xf16>
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
